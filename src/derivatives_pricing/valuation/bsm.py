"""Black-Scholes-Merton option valuation with continuous dividend yield."""

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from scipy.stats import norm
from ..utils import calculate_year_fraction, pv_discrete_dividends
from ..enums import OptionType
from ..exceptions import ValidationError

if TYPE_CHECKING:
    from .core import OptionValuation, UnderlyingData


class _BSMInputs(NamedTuple):
    """Pre-computed inputs shared across all BSM Greek calculations."""

    spot: float
    strike: float
    volatility: float
    time_to_maturity: float
    df_r: float
    df_q: float
    d1: float
    d2: float


class _BSMValuationBase:
    """Base class for Black-Scholes-Merton option valuation."""

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying: UnderlyingData = valuation_ctx.underlying  # type: ignore[assignment]

    def _calculate_d_values(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        df_r: float,
        df_q: float,
    ) -> tuple[float, float]:
        """Calculate d1 and d2 for BSM model.

        Parameters
        ----------
        spot
            Current spot price.
        strike
            Strike price.
        time_to_maturity
            Time to maturity in years.
        volatility
            Volatility (annualized).
        df_r
            Risk-free discount factor $P(0,T)$.
        df_q
            Dividend discount factor $D_q(0,T)$.

        Returns
        -------
        tuple[float, float]
            Pair ``(d1, d2)``.
        """
        if time_to_maturity <= 0:
            raise ValidationError("time_to_maturity must be positive")

        forward = spot * df_q / df_r
        denominator = volatility * np.sqrt(time_to_maturity)

        if denominator < 1e-300:
            # Zero (or near-zero) vol: deterministic limit.
            # d1 = d2 = +inf when forward > strike  →  N(d) = 1
            # d1 = d2 = -inf when forward < strike  →  N(d) = 0
            # d1 = d2 = 0    when forward == strike →  N(d) = 0.5
            if forward > strike:
                return np.inf, np.inf
            elif forward < strike:
                return -np.inf, -np.inf
            else:
                return 0.0, 0.0

        numerator = np.log(forward / strike) + 0.5 * volatility**2 * time_to_maturity
        d1 = numerator / denominator
        d2 = d1 - denominator

        return d1, d2

    def _effective_discount_factors(self, time_to_maturity: float) -> tuple[float, float]:
        df_r = float(self.valuation_ctx.discount_curve.df(time_to_maturity))
        dividend_curve = self.underlying.dividend_curve
        if dividend_curve is not None:
            df_q = float(dividend_curve.df(time_to_maturity))
        else:
            df_q = 1.0
        return df_r, df_q

    @staticmethod
    def _implied_rate_from_df(df: float, time_to_maturity: float) -> float:
        return -np.log(df) / time_to_maturity

    def _adjusted_spot(self) -> float:
        """Adjust spot for discrete dividends using PV of future dividends."""
        spot = float(self.underlying.initial_value)
        discrete_dividends = self.underlying.discrete_dividends
        if not discrete_dividends:
            return spot
        pv_divs = pv_discrete_dividends(
            discrete_dividends,
            curve_date=self.valuation_ctx.pricing_date,
            end_date=self.valuation_ctx.maturity,
            discount_curve=self.valuation_ctx.discount_curve,
            day_count_convention=self.valuation_ctx.day_count_convention,
            dividend_curve=self.underlying.dividend_curve,
        )
        return max(spot - pv_divs, 0.0)

    def _bsm_inputs(self) -> _BSMInputs:
        """Compute the full set of BSM inputs needed by pricing and Greeks.

        Extracts spot (adjusted for discrete dividends), strike, volatility,
        time to maturity, discount factors, and d-values in one place so each
        Greek method doesn't repeat the same boilerplate.
        """
        spot = self._adjusted_spot()
        strike = self.valuation_ctx.strike
        volatility = self.underlying.volatility
        time_to_maturity = calculate_year_fraction(
            self.valuation_ctx.pricing_date,
            self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)
        d1, d2 = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)
        return _BSMInputs(
            spot=spot,
            strike=strike,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            df_r=df_r,
            df_q=df_q,
            d1=d1,
            d2=d2,
        )


class _BSMEuropeanValuation(_BSMValuationBase):
    """Black-Scholes-Merton European option valuation."""

    def solve(self) -> float:
        """Compute the BSM option value."""
        inp = self._bsm_inputs()

        if self.valuation_ctx.option_type is OptionType.CALL:
            option_value = inp.spot * inp.df_q * norm.cdf(
                inp.d1
            ) - inp.strike * inp.df_r * norm.cdf(inp.d2)
        else:  # PUT
            option_value = inp.strike * inp.df_r * norm.cdf(
                -inp.d2
            ) - inp.spot * inp.df_q * norm.cdf(-inp.d1)

        return option_value

    def present_value(self) -> float:
        """Calculate present value using BSM formula."""
        return float(self.solve())

    def delta(self) -> float:
        """Calculate analytical delta for European option using closed-form BSM formula.

        delta = df_q * N(d1) for calls
        delta = df_q * (N(d1) - 1) for puts

        Where N() is the cumulative standard normal distribution.

        Returns
        -------
        float
            Analytical option delta.
        """
        inp = self._bsm_inputs()

        if inp.time_to_maturity <= 0:
            return 0.0

        if self.valuation_ctx.option_type is OptionType.CALL:
            return inp.df_q * norm.cdf(inp.d1)
        return inp.df_q * (norm.cdf(inp.d1) - 1)

    def gamma(self) -> float:
        """Calculate analytical gamma for European option using closed-form BSM formula.

        gamma = df_q * N'(d1) / (S * sigma * sqrt(T-t))

        Where N'() is the standard normal probability density function.

        Returns
        -------
        float
            Analytical option gamma.
        """
        inp = self._bsm_inputs()

        if inp.time_to_maturity <= 0:
            return 0.0

        n_prime_d1 = norm.pdf(inp.d1)
        return inp.df_q * n_prime_d1 / (inp.spot * inp.volatility * np.sqrt(inp.time_to_maturity))

    def vega(self) -> float:
        """Calculate analytical vega for European option using closed-form BSM formula.

        vega = S * df_q * N'(d1) * sqrt(T-t) / 100

        Where N'() is the standard normal probability density function.
        Vega is expressed as a 1% point change in volatility (hence the division by 100).

        Returns
        -------
        float
            Analytical option vega (per 1% volatility-point change).
        """
        inp = self._bsm_inputs()

        if inp.time_to_maturity <= 0:
            return 0.0

        n_prime_d1 = norm.pdf(inp.d1)
        return inp.spot * inp.df_q * n_prime_d1 * np.sqrt(inp.time_to_maturity) / 100

    def theta(self) -> float:
        """Calculate analytical theta for European option using closed-form BSM formula.

        Theta measures the time decay of the option value (rate of change with respect to time).
        Returns the change in option value for a 1-day decrease in time to maturity.

        For call:
            theta = -(S * N'(d1) * sigma * e^(-qT)) / (2 * sqrt(T))
                    - r * K * e^(-rT) * N(d2)
                    + q * S * e^(-qT) * N(d1)

        For put:
            theta = -(S * N'(d1) * sigma * e^(-qT)) / (2 * sqrt(T))
                    + r * K * e^(-rT) * N(-d2)
                    - q * S * e^(-qT) * N(-d1)

        ``r`` and ``q`` are the zero rates to maturity, ``-ln(df(T)) / T``,
        the flat-equivalent rates that price the option (as in QuantLib's
        ``BlackCalculator::theta``).  On a sloped curve other theta methods
        use other rates; see :meth:`OptionValuation.theta`.

        Returns
        -------
        float
            Analytical option theta (per day).
        """
        inp = self._bsm_inputs()

        if inp.time_to_maturity <= 0:
            return 0.0

        risk_free_rate = self._implied_rate_from_df(inp.df_r, inp.time_to_maturity)
        dividend_yield = self._implied_rate_from_df(inp.df_q, inp.time_to_maturity)

        n_prime_d1 = norm.pdf(inp.d1)

        # Common term for both call and put
        term1 = -(
            inp.spot * inp.df_q * n_prime_d1 * inp.volatility / (2 * np.sqrt(inp.time_to_maturity))
        )

        if self.valuation_ctx.option_type is OptionType.CALL:
            term2 = -risk_free_rate * inp.strike * inp.df_r * norm.cdf(inp.d2)
            term3 = dividend_yield * inp.spot * inp.df_q * norm.cdf(inp.d1)
            theta_annual = term1 + term2 + term3
        else:  # PUT
            term2 = risk_free_rate * inp.strike * inp.df_r * norm.cdf(-inp.d2)
            term3 = -dividend_yield * inp.spot * inp.df_q * norm.cdf(-inp.d1)
            theta_annual = term1 + term2 + term3

        # Convert from annual to per-day (divide by 365)
        return theta_annual / 365

    def rho(self) -> float:
        """Calculate analytical rho for European option using closed-form BSM formula.

        Rho measures sensitivity to interest rate changes.
        Returns the change in option value for a 1% (0.01) change in the risk-free rate.

        For call: rho = K * T * e^(-rT) * N(d2) / 100
        For put:  rho = -K * T * e^(-rT) * N(-d2) / 100

        Returns
        -------
        float
            Analytical option rho (per 1% interest-rate change).
        """
        inp = self._bsm_inputs()

        if inp.time_to_maturity <= 0:
            return 0.0

        if self.valuation_ctx.option_type is OptionType.CALL:
            return inp.strike * inp.time_to_maturity * inp.df_r * norm.cdf(inp.d2) / 100
        return -inp.strike * inp.time_to_maturity * inp.df_r * norm.cdf(-inp.d2) / 100
