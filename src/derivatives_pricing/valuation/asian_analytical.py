"""Analytical (closed-form) Asian option valuation.

Implements two analytical pricing approaches for Asian options:

1. **Geometric average** — Kemna & Vorst (1990) exact closed-form.
   Under GBM the geometric average is lognormal, so the price reduces
   to a BSM-style calculation with adjusted volatility and growth rate.

2. **Arithmetic average** — Turnbull & Wakeman (1991) moment-matching
   approximation as presented in Hull Chapter 26.  The arithmetic average
   is *not* lognormal, but its first two moments can be computed exactly.
   A lognormal distribution is fitted to those moments and Black's model
   is applied.  This is Hull's recommended approach (Section 26.13).

Current scope
-------------
- European average-price Asian call/put (geometric & arithmetic)
- N equally spaced observation dates over [t_start, T]
- Continuous dividend yield via dividend_curve

References
----------
Kemna, A. G. Z. and Vorst, A. C. F. (1990). "A Pricing Method for Options
Based on Average Asset Values", *Journal of Banking & Finance*, 14, 113–129.

Turnbull, S. M. and Wakeman, L. M. (1991). "A Quick Algorithm for Pricing
European Average Options", *Journal of Financial and Quantitative Analysis*,
26, 377–389.

Hull, J. C. *Options, Futures, and Other Derivatives*, Chapter 26.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace as dc_replace

import logging

import numpy as np
from scipy.stats import norm

from ..enums import AsianAveraging, OptionType
from ..exceptions import (
    UnsupportedFeatureError,
    ValidationError,
)
from ..utils import calculate_year_fraction

if TYPE_CHECKING:
    from .core import AsianSpec, OptionValuation, UnderlyingData


logger = logging.getLogger(__name__)


def _asian_geometric_analytical(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
    num_observations: int,
    averaging_start: float = 0.0,
) -> float:
    """Kemna-Vorst closed-form price for a geometric average-price Asian option.

    The geometric average G = (∏ S(tᵢ))^(1/M) of GBM prices is lognormal.
    Given ``num_observations = M`` equally spaced observation points,
    the average is taken over M prices at tᵢ = t_s + i·Δ
    (i = 0, 1, …, N) where N = M - 1 and Δ = (T − t_s)/N.
    The observation at
    t₀ = t_s includes the current spot price S₀, matching the convention
    used by the binomial and Monte Carlo engines.

        E[ln G] = ln S₀ + (r − q − σ²/2) · t̄
        Var[ln G] = σ² · [t_s + Δ·N·(2N+1) / (6·M)]

    where t̄ = t_s + N·Δ/2 is the mean observation time and M = N + 1.

    The option price is then the standard Black-Scholes formula applied to the
    lognormal variable G.

    Parameters
    ----------
    spot : float
        Current underlying price S₀
    strike : float
        Strike price K
    time_to_maturity : float
        Time to maturity T in years (> 0)
    volatility : float
        Annualised volatility σ (> 0)
    risk_free_rate : float
        Continuously compounded risk-free rate r
    dividend_yield : float
        Continuously compounded dividend yield q
    option_type : OptionType
        CALL or PUT
    num_observations : int
        Number of equally spaced observation points M (≥ 2), including S_t at t_s.
    averaging_start : float
        Year fraction from pricing date to the start of the averaging window
        (default 0.0 = averaging starts at pricing date)

    Returns
    -------
    float
        Present value of the geometric Asian option
    """
    if time_to_maturity <= 0:
        raise ValidationError("time_to_maturity must be positive")
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if num_observations < 2:
        raise ValidationError("num_observations must be >= 2")
    if strike < 0:
        raise ValidationError("strike must be >= 0")
    if averaging_start < 0:
        raise ValidationError("averaging_start must be >= 0")
    if averaging_start >= time_to_maturity:
        raise ValidationError("averaging_start must be < time_to_maturity")

    T = time_to_maturity
    M = num_observations  # total observation prices (including S0 at t_s)
    N = M - 1  # number of time steps
    sigma = volatility
    r = risk_free_rate
    q = dividend_yield
    S0 = spot
    K = strike
    t_s = averaging_start
    delta_t = (T - t_s) / N

    # Mean observation time: t̄ = t_s + N·Δ/2
    t_bar = t_s + N * delta_t / 2.0

    # First moment: E[ln G]
    M1 = np.log(S0) + (r - q - 0.5 * sigma**2) * t_bar

    # Second moment: Var[ln G]
    # Cov(ln S(tᵢ), ln S(tⱼ)) = σ² min(tᵢ, tⱼ) where tᵢ = t_s + i·Δ.
    # (1/M²)·ΣΣ min(tᵢ,tⱼ) for i,j ∈ {0,...,N}
    #   = t_s + Δ·N·(2N+1) / (6·M)
    # Note: the i=0, j=0 term contributes t_s (not 0), which is why the
    # t_s summand persists even though min(0,0)=0 in the *index* sum.
    M2 = sigma**2 * (t_s + delta_t * N * (2 * N + 1) / (6.0 * M))

    # Forward of geometric average: E[G] = exp(M₁ + M₂/2)
    F_G = np.exp(M1 + 0.5 * M2)

    # Discount to present
    df = np.exp(-r * T)

    # Edge case: K = 0 → deep ITM, value is just discounted forward of average
    if K == 0.0:
        if option_type is OptionType.CALL:
            return float(df * F_G)
        return 0.0

    # d-values (Black-Scholes on G)
    vol_sqrt = np.sqrt(M2)
    d1 = (np.log(F_G / K) + 0.5 * M2) / vol_sqrt
    d2 = d1 - vol_sqrt

    if option_type is OptionType.CALL:
        return float(df * (F_G * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(df * (K * norm.cdf(-d2) - F_G * norm.cdf(-d1)))


# ── Arithmetic average-price Asian (Turnbull-Wakeman / Hull §26.13) ──


def _asian_arithmetic_analytical(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
    num_observations: int,
    averaging_start: float = 0.0,
) -> float:
    """Turnbull-Wakeman moment-matching price for an arithmetic average Asian option.

    The arithmetic average S_avg = (1/M) Σ S(tᵢ) is **not** lognormal, but
    its first two moments can be computed exactly under GBM.  A lognormal
    distribution is fitted to those moments and Black's model is applied.

    Given ``num_observations = M`` equally spaced observation points,
    the average is taken over M prices at tᵢ = t_s + i·Δ  (i = 0, 1, …, N)
    where N = M - 1 and Δ = (T − t_s)/N.  The observation at t₀ = t_s includes the
    current spot price S₀, matching the convention used by the binomial
    and Monte Carlo engines.

    Moment formulas (Hull equations 26.3–26.4 for discrete observations)
    --------------------------------------------------------------------
    Forward price at tᵢ:  Fᵢ = S₀ · exp((r − q) · tᵢ)

        M₁ = E[S_avg] = (1/M) Σᵢ Fᵢ

        M₂ = E[S_avg²] = (1/M²) Σᵢ Σⱼ Fᵢ Fⱼ exp(σ² min(tᵢ, tⱼ))

    The approximate lognormal parameters are then:

        F₀ = M₁      (forward of the average)
        σ_a² = (1/T) ln(M₂ / M₁²)

    and the option price is computed via Black's model.

    Parameters
    ----------
    spot : float
        Current underlying price S₀
    strike : float
        Strike price K
    time_to_maturity : float
        Time to maturity T in years (> 0)
    volatility : float
        Annualised volatility σ (> 0)
    risk_free_rate : float
        Continuously compounded risk-free rate r
    dividend_yield : float
        Continuously compounded dividend yield q
    option_type : OptionType
        CALL or PUT
    num_observations : int
        Number of equally spaced observation points M (≥ 2), including S_t at t_s.
    averaging_start : float
        Year fraction from pricing date to the start of the averaging window
        (default 0.0 = averaging starts at pricing date)

    Returns
    -------
    float
        Present value of the arithmetic Asian option (approximate)
    """
    if time_to_maturity <= 0:
        raise ValidationError("time_to_maturity must be positive")
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if num_observations < 2:
        raise ValidationError("num_observations must be >= 2")
    if strike < 0:
        raise ValidationError("strike must be >= 0")
    if averaging_start < 0:
        raise ValidationError("averaging_start must be >= 0")
    if averaging_start >= time_to_maturity:
        raise ValidationError("averaging_start must be < time_to_maturity")

    T = time_to_maturity
    M = num_observations  # total observation prices (including S0 at t_s)
    N = M - 1
    sigma = volatility
    r = risk_free_rate
    q = dividend_yield
    S0 = spot
    K = strike
    t_s = averaging_start
    delta_t = (T - t_s) / N

    # Observation times and forward prices
    t = t_s + np.arange(M, dtype=float) * delta_t  # t[i] = t_s + i·Δ
    F = S0 * np.exp((r - q) * t)  # Fᵢ = S₀ exp((r-q) tᵢ)

    # ── First moment: M₁ = E[S_avg] = (1/M) Σ Fᵢ ──
    M1 = np.mean(F)

    # ── Second moment: M₂ = E[S_avg²] ──
    # E[S(tᵢ) S(tⱼ)] = Fᵢ Fⱼ exp(σ² min(tᵢ, tⱼ))
    #
    # Efficient O(N) computation using the identity:
    #   Σᵢ Σⱼ Fᵢ Fⱼ exp(σ² min(tᵢ,tⱼ))
    #     = Σᵢ [ Fᵢ² exp(σ² tᵢ) + 2 Fᵢ exp(σ² tᵢ) Σ_{j>i} Fⱼ ]
    #     = Σᵢ  Fᵢ exp(σ² tᵢ) · [ 2 Σ_{j≥i} Fⱼ  −  Fᵢ ]
    F_cumrev = np.cumsum(F[::-1])[::-1]  # F_cumrev[i] = Σ_{j≥i} Fⱼ
    exp_sig2_t = np.exp(sigma**2 * t)
    M2 = np.sum(F * exp_sig2_t * (2.0 * F_cumrev - F)) / M**2

    # ── Adjusted lognormal volatility ──
    # σ_a² = (1/T) · ln(M₂ / M₁²)
    sigma_a_sq = np.log(M2 / M1**2) / T
    sigma_a = np.sqrt(sigma_a_sq)

    # ── Black's model with F₀ = M₁ ──
    df = np.exp(-r * T)

    # Edge case: K = 0 → deep ITM, value is just discounted first moment
    if K == 0.0:
        if option_type is OptionType.CALL:
            return float(df * M1)
        return 0.0

    vol_sqrt_T = sigma_a * np.sqrt(T)
    d1 = (np.log(M1 / K) + 0.5 * sigma_a_sq * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    df = np.exp(-r * T)

    if option_type is OptionType.CALL:
        return float(df * (M1 * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(df * (K * norm.cdf(-d2) - M1 * norm.cdf(-d1)))


class _AnalyticalAsianValuation:
    """Analytical Asian option valuation.

    Dispatched by OptionValuation when spec is AsianSpec and
    pricing_method is BSM.

    - GEOMETRIC: Kemna-Vorst (1990) exact closed-form.
    - ARITHMETIC: Turnbull-Wakeman (1991) moment-matching approximation
      (Hull Section 26.13).
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying: UnderlyingData = valuation_ctx.underlying  # type: ignore[assignment]
        self.spec: AsianSpec = valuation_ctx.spec  # type: ignore[assignment]
        spec = self.spec
        if spec.averaging not in (AsianAveraging.GEOMETRIC, AsianAveraging.ARITHMETIC):
            raise UnsupportedFeatureError(
                "Analytical (BSM) Asian pricing requires GEOMETRIC or ARITHMETIC averaging."
            )
        if spec.fixing_dates is not None:
            raise UnsupportedFeatureError(
                "Analytical (BSM) Asian pricing currently does not support fixing_dates. "
                "Provide num_observations and use equally spaced observations."
            )
        if spec.num_observations is None:
            raise ValidationError(
                "num_observations is required on AsianSpec for analytical (BSM) pricing."
            )

    def _extract_rates(self, time_to_maturity: float) -> tuple[float, float]:
        """Extract effective continuously compounded rates from discount/dividend curves."""
        df_r = float(self.valuation_ctx.discount_curve.df(time_to_maturity))
        r = -np.log(df_r) / time_to_maturity

        dividend_curve = self.underlying.dividend_curve
        if dividend_curve is not None:
            df_q = float(dividend_curve.df(time_to_maturity))
            q = -np.log(df_q) / time_to_maturity
        else:
            q = 0.0
        return r, q

    def solve(self) -> float:
        """Return the analytical option value."""
        return self.present_value()

    def present_value(self) -> float:
        """Compute the analytical Asian option price.

        Dispatches to the Kemna-Vorst formula (geometric) or the
        Turnbull-Wakeman moment-matching approximation (arithmetic).

        For seasoned Asians (``observed_average`` set), applies Hull's K*
        strike-adjustment reduction before calling the fresh-Asian formula.
        This decomposition is only valid for arithmetic averaging;
        geometric seasoned Asians must use BINOMIAL or MONTE_CARLO.
        """
        spec = self.spec

        # ── Seasoned Asian: Hull K* strike-adjustment ────────────────────
        if spec.observed_average is not None and spec.observed_count is not None:
            return self._seasoned_pv()

        return self._fresh_pv()

    def _fresh_pv(self, spec: AsianSpec | None = None) -> float:
        """Price a fresh (non-seasoned) Asian analytically."""
        if spec is None:
            spec = self.spec
        spot = float(self.underlying.initial_value)
        strike = float(spec.strike)
        volatility = float(self.underlying.volatility)

        time_to_maturity = calculate_year_fraction(
            self.valuation_ctx.pricing_date,
            self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )

        if self.underlying.discrete_dividends:
            raise UnsupportedFeatureError(
                "Analytical Asian formula does not support discrete dividends. "
                "Use MONTE_CARLO or BINOMIAL."
            )

        r, q = self._extract_rates(time_to_maturity)

        # Determine averaging start
        averaging_start_frac = 0.0
        if (
            spec.averaging_start is not None
            and spec.averaging_start > self.valuation_ctx.pricing_date
        ):
            averaging_start_frac = calculate_year_fraction(
                self.valuation_ctx.pricing_date,
                spec.averaging_start,
                day_count_convention=self.valuation_ctx.day_count_convention,
            )

        pricer = (
            _asian_geometric_analytical
            if spec.averaging is AsianAveraging.GEOMETRIC
            else _asian_arithmetic_analytical
        )

        return pricer(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            risk_free_rate=r,
            dividend_yield=q,
            option_type=self.valuation_ctx.option_type,
            num_observations=spec.num_observations,
            averaging_start=averaging_start_frac,
        )

    def _seasoned_pv(self) -> float:
        """Price a seasoned Asian using Hull's adjusted-strike reduction.

        When n₁ fixings have already been observed with average S̄, the
        payoff of an average-price call is::

            max((n₁·S̄ + n₂·S_avg_future) / (n₁+n₂) − K, 0)

        which equals ``(n₂/(n₁+n₂)) · max(S_avg_future − K*, 0)`` where::

            K* = ((n₁+n₂)/n₂) · K  −  (n₁/n₂) · S̄

        When K* > 0 this is a newly-issued Asian with strike K* scaled by
        n₂/(n₁+n₂).  When K* ≤ 0 the option is certain to be exercised
        and its value is that of a forward contract on the remaining average.

        .. note::
           This decomposition is only valid for **arithmetic** averaging.
           Geometric seasoned Asians require BINOMIAL or MONTE_CARLO
           engines which fold past observations directly into running
           averages.

        See Hull, *Options, Futures, and Other Derivatives*, Section 26.13.
        """
        spec = self.spec
        assert spec.observed_average is not None and spec.observed_count is not None

        if spec.averaging is AsianAveraging.GEOMETRIC:
            raise UnsupportedFeatureError(
                "Hull's K* strike-adjustment is only valid for arithmetic averaging. "
                "Use PricingMethod BINOMIAL or MONTE_CARLO for seasoned geometric Asians."
            )

        n1 = spec.observed_count
        n2 = spec.num_observations  # future observations
        n_total = n1 + n2
        S_bar = spec.observed_average
        K = spec.strike

        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        scale = n2 / n_total

        logger.debug(
            "Seasoned Asian: n1=%d n2=%d S_bar=%.4f K=%.4f K*=%.4f scale=%.4f",
            n1,
            n2,
            S_bar,
            K,
            K_star,
            scale,
        )

        if K_star > 0.0:
            fresh_spec = dc_replace(spec, strike=K_star, observed_average=None, observed_count=None)
            return scale * self._fresh_pv(fresh_spec)

        # K* <= 0: option is certain to be exercised → value as forward.
        time_to_maturity = calculate_year_fraction(
            self.valuation_ctx.pricing_date,
            self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )
        df = float(self.valuation_ctx.discount_curve.df(time_to_maturity))

        # Zero-strike call = discounted E[S_avg] = discounted M₁
        zero_spec = dc_replace(
            spec,
            strike=0.0,
            option_type=OptionType.CALL,
            observed_average=None,
            observed_count=None,
        )
        disc_M1 = self._fresh_pv(zero_spec)

        if spec.option_type is OptionType.CALL:
            return scale * (disc_M1 - K_star * df)
        # Put with K*<=0: max(K* - S_avg, 0) = 0 when K*<=0 and S_avg>0
        return 0.0
