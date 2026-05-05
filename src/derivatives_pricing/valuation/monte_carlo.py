"""Monte Carlo Simulation option valuation implementations."""

from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
import logging
import datetime as dt
import numpy as np
import pandas as pd

from ..utils import calculate_year_fraction, log_timing
from ..stochastic_processes import PathSimulation, GBMProcess
from ..enums import (
    AsianAveraging,
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    OptionType,
    RebateTiming,
)
from ..exceptions import (
    NumericalError,
    UnsupportedFeatureError,
    ValidationError,
)
from .params import MonteCarloParams


if TYPE_CHECKING:
    from .core import OptionValuation, AsianSpec, BarrierSpec


logger = logging.getLogger(__name__)


# Year-length denominator for each day-count convention (used for sub-second tolerances).
_YEAR_DAYS: dict[DayCountConvention, float] = {
    DayCountConvention.ACT_360: 360.0,
    DayCountConvention.ACT_365F: 365.0,
    DayCountConvention.ACT_365_25: 365.25,
    DayCountConvention.THIRTY_360_US: 360.0,
}


def _warn_if_high_std_error(
    *,
    pv_pathwise: np.ndarray,
    pv_mean: float,
    params: MonteCarloParams,
    label: str,
) -> None:
    """Emit a warning log if MC standard error is high relative to the PV estimate."""
    if params.std_error_warn_ratio is None:
        return
    n_paths = pv_pathwise.size
    if n_paths < 2:
        return
    std_error = float(np.std(pv_pathwise, ddof=1) / np.sqrt(n_paths))
    scale = max(abs(pv_mean), 1.0e-12)
    ratio = std_error / scale
    logger.debug(
        "MC %s std_error=%.6g ratio=%.6g paths=%d",
        label,
        std_error,
        ratio,
        n_paths,
    )
    if ratio > params.std_error_warn_ratio:
        logger.warning(
            "MC %s standard error high: std_error=%.6g ratio=%.6g (>%.3g) paths=%d",
            label,
            std_error,
            ratio,
            params.std_error_warn_ratio,
            n_paths,
        )


def _resolve_time_index(
    time_grid: np.ndarray,
    target,
    label: str,
    day_count_convention: DayCountConvention,
) -> int:
    """Return the index of the entry in *time_grid* closest to *target*.

    Uses nearest-match on year-fraction offsets so that minor datetime
    rounding (for example microseconds from ``pd.date_range``) does not
    cause a lookup failure. Raises ``ValidationError`` if the nearest
    grid point is more than ~1 second away under the supplied day-count
    convention.
    """
    if len(time_grid) == 0:
        raise ValidationError(f"{label}: time_grid is empty.")

    # Fast path: exact object match (covers the common case)
    exact = np.where(time_grid == target)[0]
    if exact.size > 0:
        return int(exact[0])

    # Tolerance-based fallback: compare as year fractions from grid[0]
    ref = time_grid[0]
    target_yf = calculate_year_fraction(
        ref,
        target,
        day_count_convention=day_count_convention,
    )
    grid_yf = np.array(
        [
            calculate_year_fraction(
                ref,
                t,
                day_count_convention=day_count_convention,
            )
            for t in time_grid
        ],
        dtype=float,
    )
    idx = int(np.argmin(np.abs(grid_yf - target_yf)))
    # ~1 second tolerance expressed as year fraction.
    if abs(grid_yf[idx] - target_yf) > 1.0 / (_YEAR_DAYS[day_count_convention] * 86400):
        raise ValidationError(f"{label} not found in underlying time_grid.")
    return idx


def _vanilla_payoff(option_type: OptionType, strike: float, spot: np.ndarray) -> np.ndarray:
    """Vectorized vanilla payoff: max(S-K,0) for calls, max(K-S,0) for puts."""
    if option_type is OptionType.CALL:
        return np.maximum(spot - strike, 0.0)
    if option_type is OptionType.PUT:
        return np.maximum(strike - spot, 0.0)
    raise ValidationError(f"Unsupported option_type for vanilla payoff: {option_type}")


# ---------------------------------------------------------------------------
# Laguerre basis + ridge regression for Longstaff-Schwartz
# ---------------------------------------------------------------------------


def _laguerre_basis(x: np.ndarray, deg: int) -> np.ndarray:
    """Build a Laguerre polynomial design matrix of shape ``(n, deg+1)``.

    Uses the standard (physicists') Laguerre recurrence::

        L_0(x) = 1
        L_1(x) = 1 - x
        L_{k+1}(x) = ((2k + 1 - x) L_k(x) - k L_{k-1}(x)) / (k + 1)

    These form an orthogonal basis on [0, inf) w.r.t. e^{-x}, which
    provides better conditioning than raw power polynomials in spot.
    """
    x = np.asarray(x, dtype=float)
    cols: list[np.ndarray] = [np.ones_like(x)]
    if deg >= 1:
        cols.append(1.0 - x)
    for k in range(1, deg):
        cols.append(((2 * k + 1 - x) * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.column_stack(cols)


def _ridge_lsm_continuation(
    S_t: np.ndarray,
    Y: np.ndarray,
    itm: np.ndarray,
    strike: float | None,
    deg: int,
    ridge_lambda: float,
    min_itm: int,
) -> np.ndarray:
    """Robust continuation-value estimate for Longstaff-Schwartz.

    Regresses discounted next-step values onto a Laguerre polynomial basis
    in moneyness ``S/K`` using ridge (Tikhonov) regularisation.

    When too few ITM paths are available for a stable regression the
    continuation value falls back to the discounted next-step value
    (path-wise), which is the conservative "do no harm" default.

    Parameters
    ----------
    S_t : np.ndarray
        Spot prices at this time step for all paths.
    Y : np.ndarray
        Discounted next-step values for all paths.
    itm : np.ndarray
        Boolean mask indicating in-the-money paths.
    strike : float | None
        Option strike price (used to compute moneyness ``S/K``).
        When ``None`` (custom payoff), the mean ITM spot is used as the
        normaliser so the Laguerre basis inputs remain well-scaled.
    deg : int
        Laguerre polynomial degree.
    ridge_lambda : float
        Ridge regularisation parameter (>= 0).
    min_itm : int
        Minimum ITM paths required for regression; fewer triggers fallback.

    Returns
    -------
    np.ndarray
        Estimated continuation value for every path (zero for OTM paths).
    """
    cont = np.zeros_like(S_t, dtype=float)
    if not np.any(itm):
        return cont

    S_itm = S_t[itm]
    Y_itm = Y[itm]
    n = S_itm.size

    # Too few ITM points for a stable fit — use cross-sectional mean
    # (degree-0 regression) to avoid path-wise foresight bias.
    if n < max(min_itm, deg + 1):
        cont[itm] = np.mean(Y_itm)
        return cont

    normaliser = strike if strike is not None else max(float(np.mean(S_itm)), 1e-12)
    x = S_itm / normaliser  # moneyness, always positive
    X = _laguerre_basis(x, deg=deg)
    cont[itm] = _ridge_fit_predict(X, Y_itm, X, ridge_lambda)
    return cont


def _ridge_fit_predict(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_pred: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """Fit ridge regression on ``(X_fit, y_fit)`` and predict on ``X_pred``."""
    p = X_fit.shape[1]
    XtX = X_fit.T @ X_fit
    Xty = X_fit.T @ y_fit
    beta = np.linalg.solve(XtX + ridge_lambda * np.eye(p), Xty)
    return X_pred @ beta


def _year_fractions(
    pricing_date,
    dates: np.ndarray,
    day_count_convention: DayCountConvention,
) -> np.ndarray:
    return np.array(
        [
            calculate_year_fraction(
                pricing_date,
                d,
                day_count_convention=day_count_convention,
            )
            for d in dates
        ],
        dtype=float,
    )


def _is_pricing_time(
    pricing_date,
    current_time,
    day_count_convention: DayCountConvention,
) -> bool:
    """Return whether *current_time* is effectively equal to *pricing_date*."""
    t = calculate_year_fraction(
        pricing_date,
        current_time,
        day_count_convention=day_count_convention,
    )
    return abs(t) <= 1.0 / (_YEAR_DAYS[day_count_convention] * 86400)


def _time_zero_lsm_value(
    discounted_continuation: np.ndarray,
    *,
    immediate_exercise: float | None,
) -> np.ndarray:
    """Return the time-zero LSM value.

    At t=0 the spot is identical across all paths, so the immediate intrinsic
    is a scalar and the LSM continuation estimate is the *mean* of discounted
    next-step values.  The exercise decision compares those two scalars.

    - Exercise → all paths take the constant intrinsic.
    - Don't exercise → each path keeps its own discounted next-step value
    """
    if immediate_exercise is None:
        return discounted_continuation.astype(float, copy=True)
    continuation_0 = float(np.mean(discounted_continuation))
    if immediate_exercise > continuation_0:
        return np.full_like(discounted_continuation, immediate_exercise, dtype=float)
    return discounted_continuation.astype(float, copy=True)


class _MCValuationBase:
    """Common base for all Monte Carlo valuation engines."""

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        assert isinstance(valuation_ctx.params, MonteCarloParams)
        self.mc_params = valuation_ctx.params
        assert isinstance(valuation_ctx.underlying, PathSimulation)
        self.underlying = valuation_ctx.underlying


class _MCEuropeanValuation(_MCValuationBase):
    """Implementation of European option valuation using Monte Carlo."""

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector at maturity (one value per path)."""
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid

        time_index_end = _resolve_time_index(
            time_grid,
            self.valuation_ctx.maturity,
            "maturity",
            day_count_convention=self.underlying.day_count_convention,
        )
        maturity_value = paths[time_index_end]

        K = self.valuation_ctx.strike
        if self.valuation_ctx.option_type in (OptionType.CALL, OptionType.PUT):
            return _vanilla_payoff(self.valuation_ctx.option_type, K, maturity_value)

        assert hasattr(self.valuation_ctx.spec, "payoff")
        return self.valuation_ctx.spec.payoff(maturity_value)  # type: ignore[union-attr]

    def present_value(self) -> float:
        """Return the scalar present value."""
        with log_timing(logger, "MC European present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC European paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="European",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = self.valuation_ctx._maturity_year_fraction()
        discount_factor = float(self.valuation_ctx.discount_curve.df(ttm))
        return discount_factor * payoff_vector

    # ------------------------------------------------------------------
    # MC Greeks — pathwise and likelihood-ratio estimators
    # ------------------------------------------------------------------

    def _simulate_terminal(self) -> tuple[np.ndarray, int, float, float]:
        """Simulate paths and return terminal prices with key scalars.

        Returns
        -------
        (ST, idx, ttm, df) — terminal spot array, time-grid index of
        maturity, time to maturity in years, and risk-free discount factor.
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        idx = _resolve_time_index(
            time_grid,
            self.valuation_ctx.maturity,
            "maturity",
            day_count_convention=self.underlying.day_count_convention,
        )
        ST: np.ndarray = paths[idx]
        ttm = self.valuation_ctx._maturity_year_fraction()
        df = float(self.valuation_ctx.discount_curve.df(ttm))
        return ST, idx, ttm, df

    def _effective_terminal_z(self, idx: int, ttm: float) -> np.ndarray:
        r"""Compute the effective terminal standard normal from cached per-step normals.

        .. math::
            Z_{\text{eff}}
            = \frac{\sum_i \sqrt{\Delta t_i}\,Z_i}{\sqrt{T}}

        Must be called **after** :meth:`_simulate_terminal` (which populates
        :pyattr:`PathSimulation.last_normals`).
        """
        z_all = self.underlying.last_normals  # (num_steps, num_paths)
        if z_all is None:
            raise ValidationError(
                "Internal state error: missing cached MC shocks (last_normals). "
                "Call _simulate_terminal() before _effective_terminal_z()."
            )
        time_deltas = self.underlying._time_deltas()  # (num_steps,)
        sqrt_dt = np.sqrt(time_deltas[:idx])

        if not np.isclose(time_deltas[:idx].sum(), ttm):
            raise NumericalError(
                f"Time deltas sum {time_deltas[:idx].sum()} does not match time to maturity {ttm}"
            )
        return sqrt_dt @ z_all[:idx] / np.sqrt(ttm)

    # --- pathwise Greeks --------------------------------------------------

    def delta_pathwise(self) -> float:
        r"""Pathwise (IPA) delta estimator.

        Call: :math:`\\Delta = e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T>K}\\,S_T/S_0]`

        Put:  :math:`\\Delta = -e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T<K}\\,S_T/S_0]`
        """
        ST, _idx, _ttm, df = self._simulate_terminal()
        S0 = float(self.underlying.initial_value)
        K = self.valuation_ctx.strike
        if self.valuation_ctx.option_type is OptionType.CALL:
            return float(np.mean(df * (ST > K) * (ST / S0)))
        return float(np.mean(-df * (ST < K) * (ST / S0)))

    def vega_pathwise(self) -> float:
        r"""Pathwise (IPA) vega estimator (per 1 pp change in vol).

        Uses :math:`\\partial S_T/\\partial\\sigma = S_T(-\\sigma T + \\sqrt{T}\\,Z)`.

        Call: :math:`\\nu = e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T>K}\\,S_T(-\\sigma T+\\sqrt{T}Z)]`

        Put:  :math:`\\nu = -e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T<K}\\,S_T(-\\sigma T+\\sqrt{T}Z)]`
        """
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        sigma = float(self.underlying.volatility)
        K = self.valuation_ctx.strike
        dST_dsigma = ST * (-sigma * ttm + np.sqrt(ttm) * Z)
        if self.valuation_ctx.option_type is OptionType.CALL:
            return float(np.mean(df * (ST > K) * dST_dsigma)) / 100
        return float(np.mean(-df * (ST < K) * dST_dsigma)) / 100

    # --- likelihood-ratio Greeks ------------------------------------------

    @staticmethod
    def _require_positive_sigma_for_lr(sigma: float, greek: str) -> None:
        """LR estimators divide by σ; the lognormal density is degenerate at σ=0."""
        if sigma <= 0.0:
            raise UnsupportedFeatureError(
                f"Likelihood-ratio {greek} is undefined at sigma=0 (the lognormal "
                "density degenerates to a Dirac at the forward, so the score "
                "function diverges). Use GreekCalculationMethod.NUMERICAL or "
                "PATHWISE instead, or call BSM analytical greeks."
            )

    def delta_lr(self) -> float:
        r"""Likelihood-ratio (score-function) delta estimator.

        :math:`\\Delta = e^{-rT}\\,\\mathbb{E}\\!\\left[\\Phi(S_T)\\,
        \\frac{Z}{\\sigma\\sqrt{T}\\,S_0}\\right]`
        """
        sigma = float(self.underlying.volatility)
        self._require_positive_sigma_for_lr(sigma, "delta")
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        S0 = float(self.underlying.initial_value)
        K = self.valuation_ctx.strike
        payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, ST)
        return float(np.mean(df * payoff * Z / (sigma * np.sqrt(ttm) * S0)))

    def vega_lr(self) -> float:
        r"""Likelihood-ratio (score-function) vega estimator (per 1 pp).

        The score of the lognormal density w.r.t. :math:`\\sigma` is

        .. math::
            \\frac{\\partial\\ln p}{\\partial\\sigma}
            = \\frac{Z^2 - 1}{\\sigma} \\;-\\; Z\\sqrt{T}

        The second term arises because the risk-neutral drift
        :math:`\\mu = r - q - \\tfrac12\\sigma^2` depends on :math:`\\sigma`
        via the Itô correction.
        """
        sigma = float(self.underlying.volatility)
        self._require_positive_sigma_for_lr(sigma, "vega")
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        K = self.valuation_ctx.strike
        payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, ST)
        score = (Z**2 - 1) / sigma - Z * np.sqrt(ttm)
        return float(np.mean(df * payoff * score)) / 100

    # --- theta estimators -------------------------------------------------

    def _risk_free_and_div_rates(self, ttm: float) -> tuple[float, float]:
        """Extract annualised risk-free rate *r* and dividend yield *q*."""
        r = -np.log(float(self.valuation_ctx.discount_curve.df(ttm))) / ttm
        div_curve = self.underlying.dividend_curve
        q = -np.log(float(div_curve.df(ttm))) / ttm if div_curve is not None else 0.0
        return float(r), float(q)

    def theta_pathwise(self) -> float:
        r"""Pathwise (IPA) theta estimator (per calendar day).

        .. math::
            \Theta_{\text{PW}} = -e^{-rT}\,\mathbb{E}\!\left[
            \Phi'(S_T)\,S_T\!\left(a + \frac{\sigma Z}{2\sqrt{T}}\right)
            - r\,\Phi(S_T)\right]

        where :math:`a = r - q - \tfrac12\sigma^2`.
        """
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        sigma = float(self.underlying.volatility)
        K = self.valuation_ctx.strike
        r, q = self._risk_free_and_div_rates(ttm)
        a = r - q - 0.5 * sigma**2

        payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, ST)
        if self.valuation_ctx.option_type is OptionType.CALL:
            indicator = (ST > K).astype(float)
        else:
            indicator = -(ST < K).astype(float)

        sensitivity = indicator * ST * (a + sigma * Z / (2 * np.sqrt(ttm)))
        per_year = -df * (sensitivity - r * payoff)
        return float(np.mean(per_year)) / 365

    def theta_lr(self) -> float:
        r"""Likelihood-ratio (score-function) theta estimator (per calendar day).

        .. math::
            \Theta_{\text{LR}} = -e^{-rT}\,\mathbb{E}\!\left[
            \Phi(S_T)\!\left(\frac{Z^2-1}{2T}
            + \frac{aZ}{\sigma\sqrt{T}} - r\right)\right]

        where :math:`a = r - q - \tfrac12\sigma^2`.
        """
        sigma = float(self.underlying.volatility)
        self._require_positive_sigma_for_lr(sigma, "theta")
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        K = self.valuation_ctx.strike
        r, q = self._risk_free_and_div_rates(ttm)
        a = r - q - 0.5 * sigma**2

        payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, ST)
        score = (Z**2 - 1) / (2 * ttm) + a * Z / (sigma * np.sqrt(ttm)) - r
        per_year = -df * payoff * score
        return float(np.mean(per_year)) / 365

    # --- rho estimators ----------------------------------------------------

    def rho_pathwise(self) -> float:
        r"""Pathwise (IPA) rho estimator (per 1 % parallel rate move).

        Call: :math:`\rho = \mathrm{df}\,T\,K\,\mathbb{E}[\mathbf{1}_{S_T>K}]`

        Put:  :math:`\rho = -\mathrm{df}\,T\,K\,\mathbb{E}[\mathbf{1}_{S_T<K}]`
        """
        ST, _idx, ttm, df = self._simulate_terminal()
        K = self.valuation_ctx.strike
        if self.valuation_ctx.option_type is OptionType.CALL:
            return float(np.mean(df * ttm * K * (ST > K))) * 0.01
        return float(np.mean(-df * ttm * K * (ST < K))) * 0.01

    def rho_lr(self) -> float:
        r"""Likelihood-ratio (score-function) rho estimator (per 1 % rate move).

        .. math::
            \rho_{\text{LR}} = \mathrm{df}\,\mathbb{E}\!\left[
            \Phi(S_T)\!\left(\frac{\sqrt{T}}{\sigma}\,Z - T\right)\right]
        """
        sigma = float(self.underlying.volatility)
        self._require_positive_sigma_for_lr(sigma, "rho")
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        K = self.valuation_ctx.strike
        payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, ST)
        score = np.sqrt(ttm) / sigma * Z - ttm
        return float(np.mean(df * payoff * score)) * 0.01

    # --- finite-difference of pathwise delta ------------------------------

    def gamma_pathwise_fd(self, epsilon: float | None = None) -> float:
        r"""Gamma via central difference of pathwise deltas on one simulation.

        Under GBM, :math:`S_T` scales linearly with :math:`S_0`, so bumped
        terminal values are obtained analytically from a single set of paths:

        .. math::
            S_T^{\pm} = S_T \cdot (S_0 \pm h) / S_0

        The pathwise delta at each bumped spot is computed on these
        rescaled terminals (same random draws), and gamma follows from
        the standard central-difference formula:

        .. math::
            \Gamma \approx
            \frac{\Delta_{\text{pw}}(S_0+h) - \Delta_{\text{pw}}(S_0-h)}{2h}

        Parameters
        ----------
        epsilon
            Spot bump size ``h`` for central differences. If ``None``, uses
            ``S0 / 100``.

        Returns
        -------
        float
            Gamma estimate.
        """
        ST, _idx, _ttm, df = self._simulate_terminal()
        S0 = float(self.underlying.initial_value)
        K = self.valuation_ctx.strike
        is_call = self.valuation_ctx.option_type is OptionType.CALL

        if epsilon is None:
            epsilon = S0 / 100  # 1 % of spot

        S0_up = S0 + epsilon
        S0_dn = S0 - epsilon
        ST_up = ST * (S0_up / S0)
        ST_dn = ST * (S0_dn / S0)

        if is_call:
            delta_up = float(np.mean(df * (ST_up > K) * (ST_up / S0_up)))
            delta_dn = float(np.mean(df * (ST_dn > K) * (ST_dn / S0_dn)))
        else:
            delta_up = float(np.mean(-df * (ST_up < K) * (ST_up / S0_up)))
            delta_dn = float(np.mean(-df * (ST_dn < K) * (ST_dn / S0_dn)))

        return (delta_up - delta_dn) / (2 * epsilon)


class _MCAmericanValuation(_MCValuationBase):
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def solve(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate underlying paths and intrinsic payoff matrix over time.

        Returns
        -------
        tuple of (spot_paths, payoff, time_index_start, time_index_end)
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        time_index_start = _resolve_time_index(
            time_grid,
            self.valuation_ctx.pricing_date,
            "Pricing date",
            day_count_convention=self.underlying.day_count_convention,
        )
        time_index_end = _resolve_time_index(
            time_grid,
            self.valuation_ctx.maturity,
            "maturity",
            day_count_convention=self.underlying.day_count_convention,
        )

        spot_paths = paths[time_index_start : time_index_end + 1]

        K = self.valuation_ctx.strike
        if self.valuation_ctx.option_type in (OptionType.CALL, OptionType.PUT):
            payoff = _vanilla_payoff(self.valuation_ctx.option_type, K, spot_paths)
        else:
            payoff_fn = self.valuation_ctx.spec.payoff  # type: ignore[union-attr]
            payoff = payoff_fn(spot_paths)

        return spot_paths, payoff, time_index_start, time_index_end

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz regression method."""
        with log_timing(logger, "MC American present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
            self.mc_params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="American",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path (LSM output at pricing date)."""
        spot_paths, intrinsic_values, time_index_start, time_index_end = self.solve()
        times = self.underlying.time_grid[time_index_start : time_index_end + 1]
        t_grid = _year_fractions(
            self.valuation_ctx.pricing_date,
            times,
            day_count_convention=self.underlying.day_count_convention,
        )
        discount_factors = self.valuation_ctx.discount_curve.df(t_grid)
        values = np.zeros_like(intrinsic_values)
        values[-1] = intrinsic_values[-1]

        for t in range(len(times) - 2, 0, -1):
            df_step = discount_factors[t + 1] / discount_factors[t]
            itm = intrinsic_values[t] > 0

            continuation = _ridge_lsm_continuation(
                S_t=spot_paths[t],
                Y=df_step * values[t + 1],
                itm=itm,
                strike=self.valuation_ctx.strike,
                deg=self.mc_params.deg,
                ridge_lambda=self.mc_params.ridge_lambda,
                min_itm=self.mc_params.min_itm,
            )

            values[t] = np.where(
                intrinsic_values[t] > continuation,
                intrinsic_values[t],
                df_step * values[t + 1],
            )

        df0 = discount_factors[1] / discount_factors[0]
        immediate_0 = float(np.mean(intrinsic_values[0]))
        return _time_zero_lsm_value(df0 * values[1], immediate_exercise=immediate_0)


class _MCAsianBase(_MCValuationBase):
    """Shared infrastructure for European and American MC Asian valuations.

    Provides common ``__init__`` validation, fixing-date index resolution,
    payoff computation, and averaging-path extraction so that the two
    concrete subclasses only contain exercise-specific logic.
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        super().__init__(valuation_ctx)
        self.spec: AsianSpec = valuation_ctx.spec  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fixing_indices(self, time_grid: np.ndarray) -> np.ndarray | None:
        """Return grid indices for the contractual Asian observation schedule."""
        fixing_dates = self.valuation_ctx._asian_fixing_dates()
        return np.array(
            [
                _resolve_time_index(
                    time_grid,
                    d,
                    f"fixing_date: {d.strftime('%Y-%m-%d')}",
                    day_count_convention=self.underlying.day_count_convention,
                )
                for d in fixing_dates
            ],
            dtype=int,
        )

    def _asian_payoff(self, avg: np.ndarray) -> np.ndarray:
        """Asian payoff given average prices."""
        K = self.valuation_ctx.strike
        if self.spec.option_type is OptionType.CALL:
            return np.maximum(avg - K, 0.0)
        return np.maximum(K - avg, 0.0)

    def _extract_averaging_paths(
        self, paths: np.ndarray, time_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the path rows that belong to the averaging window.

        Returns
        -------
        (averaging_paths, times)
            averaging_paths : (num_fixings, n_paths) spot prices at fixing dates
            times           : (num_fixings,) datetime sub-grid for those dates
        """
        fixing_idx = self._fixing_indices(time_grid)
        return paths[fixing_idx], time_grid[fixing_idx]


class _MCAsianEuropeanValuation(_MCAsianBase):
    """Implementation of European Asian option valuation using Monte Carlo.

    Asian options are path-dependent options where the payoff depends on the average
    price of the underlying over the averaging period.
    """

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector based on path averages.

        Seasoned Asians (``spec.observed_average`` and
        ``spec.observed_count`` set) are handled directly:

        - **Arithmetic**: past observations are folded into the mean via
          ``(n₁·S̄ + Σ S_i) / (n₁ + n₂)``.
        - **Geometric**: past observations are folded via log-sum
          decomposition.

        Returns
        -------
        np.ndarray
            Payoff for each path based on the average spot price
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        averaging_paths, _ = self._extract_averaging_paths(paths, self.underlying.time_grid)

        spec = self.spec
        running_avg = _running_averages(
            averaging_paths,
            spec.averaging,
            observed_count=spec.observed_count or 0,
            observed_average=spec.observed_average,
        )
        avg_prices = running_avg[-1]

        return self._asian_payoff(avg_prices)

    def present_value(self) -> float:
        """Return the scalar present value."""
        with log_timing(logger, "MC Asian present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Asian paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Asian",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = self.valuation_ctx._maturity_year_fraction()
        discount_factor = float(self.valuation_ctx.discount_curve.df(ttm))
        return discount_factor * payoff_vector


def _running_averages(
    averaging_paths: np.ndarray,
    averaging: AsianAveraging,
    *,
    observed_count: int = 0,
    observed_average: float | None = None,
) -> np.ndarray:
    """Compute the running average at each time step along axis 0.

    Parameters
    ----------
    averaging_paths : np.ndarray, shape (num_fixing_dates, n_paths)
        Spot prices at each fixing date for every simulated path.
    averaging : AsianAveraging
        ARITHMETIC or GEOMETRIC.
    observed_count : int, default 0
        Number of already observed fixings (seasoned Asian).
    observed_average : float | None, default None
        Realized average across observed fixings. Required when
        ``observed_count > 0``.

    Returns
    -------
    np.ndarray, shape (num_fixing_dates, n_paths)
        Running average at each fixing date.
    """
    num_fixings = averaging_paths.shape[0]
    counts = np.arange(1, num_fixings + 1, dtype=float).reshape(-1, 1)
    n1 = int(observed_count)
    if n1 < 0:
        raise ValidationError("observed_count must be >= 0")
    if n1 > 0 and observed_average is None:
        raise ValidationError("observed_average must be provided when observed_count > 0")

    if averaging is AsianAveraging.ARITHMETIC:
        cumsum = np.cumsum(averaging_paths, axis=0)
        if n1 > 0:
            return (n1 * observed_average + cumsum) / (n1 + counts)
        return cumsum / counts

    if averaging is AsianAveraging.GEOMETRIC:
        if np.any(averaging_paths <= 0.0):
            raise NumericalError("Geometric averaging requires strictly positive path prices.")
        log_cumsum = np.cumsum(np.log(averaging_paths), axis=0)
        if n1 > 0:
            return np.exp((n1 * np.log(observed_average) + log_cumsum) / (n1 + counts))
        return np.exp(log_cumsum / counts)

    raise ValidationError(f"Unsupported averaging method: {averaging}")


def _asian_lsm_continuation(
    S_t: np.ndarray,
    A_t: np.ndarray,
    Y: np.ndarray,
    itm: np.ndarray,
    strike: float,
    deg: int,
    ridge_lambda: float,
    min_itm: int,
) -> np.ndarray:
    """Robust continuation-value estimate for Asian American LSM.

    Regresses discounted next-step values onto an additive Laguerre basis
    in average-moneyness ``A/K`` (primary), spot-moneyness ``S/K``
    (secondary), and a single interaction feature ``S/A`` using ridge
    regularisation.

    The design matrix has ``2*(deg+1) + 1`` columns::

        [L_0(A/K), ..., L_d(A/K),  L_0(S/K), ..., L_d(S/K),  S/A]

    This is more compact and better-conditioned than a cross-monomial
    basis in raw (S, A) which had ``(d+1)(d+2)/2`` columns with
    near-collinear terms.

    Parameters
    ----------
    S_t : np.ndarray
        Spot prices at this time step for all paths.
    A_t : np.ndarray
        Running average at this time step for all paths.
    Y : np.ndarray
        Discounted next-step values for all paths.
    itm : np.ndarray
        Boolean mask indicating in-the-money paths.
    strike : float
        Option strike price K (used to compute moneyness).
    deg : int
        Laguerre polynomial degree for each moneyness feature.
    ridge_lambda : float
        Ridge regularisation parameter (>= 0).
    min_itm : int
        Minimum ITM paths required for regression; fewer triggers fallback.

    Returns
    -------
    np.ndarray
        Estimated continuation value for every path (zero for OTM paths).
    """
    cont = np.zeros_like(S_t, dtype=float)
    if not np.any(itm):
        return cont

    S_itm = S_t[itm]
    A_itm = A_t[itm]
    Y_itm = Y[itm]
    n = S_itm.size

    # Too few ITM points — cross-sectional mean (degree-0 regression)
    # to avoid path-wise foresight bias.
    n_cols = 2 * (deg + 1) + 1
    if n < max(min_itm, n_cols):
        cont[itm] = np.mean(Y_itm)
        return cont

    # Build additive Laguerre design matrix
    avg_moneyness = A_itm / strike  # A_t / K  (primary)
    spot_moneyness = S_itm / strike  # S_t / K  (secondary)
    # Interaction: spot relative to current average  (S_t / A_t)
    interaction = S_itm / np.maximum(A_itm, 1e-12)

    L_avg = _laguerre_basis(avg_moneyness, deg=deg)  # (n, deg+1)
    L_spot = _laguerre_basis(spot_moneyness, deg=deg)  # (n, deg+1)
    X = np.column_stack([L_avg, L_spot, interaction])  # (n, 2*(deg+1)+1)

    cont[itm] = _ridge_fit_predict(X, Y_itm, X, ridge_lambda)
    return cont


class _MCAsianAmericanValuation(_MCAsianBase):
    """American Asian option via Longstaff-Schwartz on the joint (S_t, A_t) state.

    At each exercise opportunity *t* the holder can receive max(A_t − K, 0)
    (call) or max(K − A_t, 0) (put), where A_t is the running average of
    the spot observations from the averaging start up to *t*.

    The continuation value is estimated by ridge regression on an additive
    Laguerre polynomial basis in average-moneyness A_t/K (primary) and
    spot-moneyness S_t/K (secondary), plus a single S_t/A_t interaction
    feature.

    Both arithmetic and geometric averaging are supported.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_averaging_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate paths and compute running averages & intrinsic values.

        Returns
        -------
        (averaging_paths, running_avg, intrinsic, times)
            averaging_paths : (num_fixings, n_paths) spot prices at fixing dates
            running_avg     : (num_fixings, n_paths) running average at each fixing date
            intrinsic       : (num_fixings, n_paths) intrinsic payoff at each fixing date
            times           : (num_fixings,) datetime time grid for the fixing window
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        averaging_paths, times = self._extract_averaging_paths(paths, self.underlying.time_grid)

        spec = self.spec
        n1 = spec.observed_count or 0
        s_bar = spec.observed_average
        running_avg = _running_averages(
            averaging_paths,
            spec.averaging,
            observed_count=n1,
            observed_average=s_bar,
        )

        intrinsic = self._asian_payoff(running_avg)
        return averaging_paths, running_avg, intrinsic, times

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate path-level Asian option data.

        Returns
        -------
        (averaging_paths, running_avg, intrinsic_values)
            Each has shape (num_fixings, n_paths).
        """
        averaging_paths, running_avg, intrinsic, _ = self._get_averaging_data()
        return averaging_paths, running_avg, intrinsic

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz with (S, A) regression."""
        with log_timing(logger, "MC Asian American present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Asian American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
            self.mc_params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Asian American",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Discounted PV for each path via LSM on (S_t, A_t)."""
        averaging_paths, running_avg, intrinsic, times = self._get_averaging_data()

        t_grid = _year_fractions(
            self.valuation_ctx.pricing_date,
            times,
            day_count_convention=self.underlying.day_count_convention,
        )
        discount_factors = self.valuation_ctx.discount_curve.df(t_grid)

        num_fixings = averaging_paths.shape[0]
        n_paths = averaging_paths.shape[1]
        deg = self.mc_params.deg
        first_fixing_is_pricing_date = _is_pricing_time(
            self.valuation_ctx.pricing_date,
            times[0],
            self.underlying.day_count_convention,
        )

        # Terminal payoff
        values = np.zeros((num_fixings, n_paths))
        values[-1] = intrinsic[-1]

        # Backward induction
        stop = 0 if first_fixing_is_pricing_date else -1
        for t in range(num_fixings - 2, stop, -1):
            df_step = discount_factors[t + 1] / discount_factors[t]
            itm = intrinsic[t] > 0

            continuation = _asian_lsm_continuation(
                S_t=averaging_paths[t],
                A_t=running_avg[t],
                Y=df_step * values[t + 1],
                itm=itm,
                strike=self.valuation_ctx.strike,
                deg=deg,
                ridge_lambda=self.mc_params.ridge_lambda,
                min_itm=self.mc_params.min_itm,
            )

            values[t] = np.where(
                intrinsic[t] > continuation,
                intrinsic[t],
                df_step * values[t + 1],
            )

        if first_fixing_is_pricing_date:
            df0 = discount_factors[1] / discount_factors[0]
            immediate_0 = float(np.mean(intrinsic[0]))
            return _time_zero_lsm_value(df0 * values[1], immediate_exercise=immediate_0)

        return discount_factors[0] * values[0]


# ══════════════════════════════════════════════════════════════════════════
# Barrier MC helpers
# ══════════════════════════════════════════════════════════════════════════


def _brownian_bridge_hit_prob(
    S_i: np.ndarray,
    S_next: np.ndarray,
    barrier: float,
    sigma: float,
    dt_step: float,
    direction: BarrierDirection,
) -> np.ndarray:
    """Conditional probability of barrier crossing between two GBM endpoints.

    Uses the Brownian bridge formula for log-normal diffusion.  Given that
    ``log S`` is Brownian with drift, the maximum (or minimum) between two
    observed values follows the reflection-principle result.

    Parameters
    ----------
    S_i, S_next
        Spot prices at the start/end of the interval, shape ``(num_paths,)``.
    barrier
        Barrier level *H*.
    sigma
        Instantaneous volatility.
    dt_step
        Year-fraction length of the interval.
    direction
        UP or DOWN.

    Returns
    -------
    np.ndarray
        Per-path hit probabilities, shape ``(num_paths,)``, clamped to [0, 1].
    """
    p = np.zeros_like(S_i)

    # At sigma == 0 the path is deterministic and monotone, so any in-step
    # crossing must show up at an endpoint -- the `crossed` mask catches it
    # and the Brownian-bridge correction is both undefined (1/sigma**2) and
    # unnecessary (interior hit probability is 0).
    sigma_positive = sigma > 0.0

    if direction is BarrierDirection.UP:
        crossed = (S_i >= barrier) | (S_next >= barrier)
        if sigma_positive:
            both_below = ~crossed
            if np.any(both_below):
                log_a = np.log(barrier / S_i[both_below])
                log_b = np.log(barrier / S_next[both_below])
                p[both_below] = np.exp(-2.0 * log_a * log_b / (sigma**2 * dt_step))
    else:
        crossed = (S_i <= barrier) | (S_next <= barrier)
        if sigma_positive:
            both_above = ~crossed
            if np.any(both_above):
                log_a = np.log(S_i[both_above] / barrier)
                log_b = np.log(S_next[both_above] / barrier)
                p[both_above] = np.exp(-2.0 * log_a * log_b / (sigma**2 * dt_step))

    p[crossed] = 1.0
    return np.clip(p, 0.0, 1.0)


def _resolve_barrier_monitoring_indices(
    time_grid: np.ndarray,
    spec: BarrierSpec,
    pricing_date: dt.datetime,
    day_count_convention: DayCountConvention,
) -> np.ndarray:
    """Map barrier monitoring schedule to simulation time-grid indices.

    Parameters
    ----------
    time_grid
        Array of datetime objects from :class:`PathSimulation`.
    spec
        Barrier specification with monitoring mode and schedule.
    pricing_date
        Valuation date.
    day_count_convention
        Convention used by the simulation.

    Returns
    -------
    np.ndarray
        Sorted, deduplicated integer indices into *time_grid*.
    """
    if spec.monitoring is BarrierMonitoring.CONTINUOUS:
        return np.arange(len(time_grid))

    if spec.monitoring_dates is not None:
        indices = [
            _resolve_time_index(
                time_grid,
                d,
                "monitoring date",
                day_count_convention=day_count_convention,
            )
            for d in spec.monitoring_dates
        ]
    elif spec.num_observations is not None:
        # Match OptionValuation._barrier_monitoring_dates: N dates at
        # T/N, 2T/N, ..., T — pricing date excluded (BGK / Boyle-Tian
        # convention).  Build N+1 evenly-spaced dates and drop the first.
        dates = pd.date_range(
            start=pricing_date,
            end=spec.maturity,
            periods=spec.num_observations + 1,
        )[1:]
        indices = [
            _resolve_time_index(
                time_grid,
                d.to_pydatetime(),
                "monitoring date",
                day_count_convention=day_count_convention,
            )
            for d in dates
        ]
    else:
        raise ValidationError(
            "Discrete barrier monitoring requires num_observations or monitoring_dates."
        )
    return np.unique(np.array(indices, dtype=int))


def _barrier_log_distance(
    spot: np.ndarray,
    barrier: float,
    direction: BarrierDirection,
) -> np.ndarray:
    """Return positive log-distance from the absorbing barrier.

    For alive knock-out states, smaller values mean closer proximity to the
    barrier. The log transform captures the natural geometry of GBM barriers
    better than raw spot ratios.
    """
    safe_spot = np.maximum(np.asarray(spot, dtype=float), 1.0e-12)
    if direction is BarrierDirection.UP:
        return np.maximum(np.log(barrier / safe_spot), 0.0)
    return np.maximum(np.log(safe_spot / barrier), 0.0)


def _barrier_lsm_design_matrix(
    S_t: np.ndarray,
    strike: float,
    barrier: float,
    direction: BarrierDirection,
    deg: int,
) -> np.ndarray:
    """Build a barrier-aware design matrix for American knock-out LSM.

    The standard Laguerre basis in moneyness is augmented with barrier
    distance terms so the regression can better represent the sharp drop in
    continuation value near the absorbing barrier.
    """
    x = np.asarray(S_t, dtype=float) / strike
    base = _laguerre_basis(x, deg=deg)
    dist = _barrier_log_distance(S_t, barrier, direction)
    inv_dist = 1.0 / np.maximum(dist, 5.0e-2)
    near_barrier = np.exp(-4.0 * dist)
    columns = [
        base,
        dist[:, None],
        (dist**2)[:, None],
        inv_dist[:, None],
        near_barrier[:, None],
        (x * near_barrier)[:, None],
    ]
    if deg >= 2:
        columns.append((base[:, 1] * near_barrier)[:, None])
    return np.column_stack(columns)


def _ridge_barrier_lsm_continuation(
    S_t: np.ndarray,
    Y: np.ndarray,
    itm: np.ndarray,
    strike: float,
    barrier: float,
    direction: BarrierDirection,
    deg: int,
    ridge_lambda: float,
    min_itm: int,
) -> np.ndarray:
    """Barrier-aware continuation estimate for American knock-out LSM.

    Uses a barrier-distance basis plus a near/far split so the regression can
    fit the sharp continuation cliff close to the absorbing boundary without
    distorting the smoother far-from-barrier region.
    """
    cont = np.zeros_like(S_t, dtype=float)
    if not np.any(itm):
        return cont

    S_itm = S_t[itm]
    Y_itm = Y[itm]
    n = S_itm.size

    X_itm = _barrier_lsm_design_matrix(
        S_itm,
        strike=strike,
        barrier=barrier,
        direction=direction,
        deg=deg,
    )
    min_regression_paths = max(min_itm, X_itm.shape[1] + 2)
    if n < min_regression_paths:
        cont[itm] = np.mean(Y_itm)
        return cont

    dist_itm = _barrier_log_distance(S_itm, barrier, direction)
    split_threshold = min(float(np.quantile(dist_itm, 0.35)), 0.25)
    near_mask = dist_itm <= split_threshold
    far_mask = ~near_mask
    min_bucket = max(12, X_itm.shape[1] + 1)

    if np.sum(near_mask) < min_bucket or np.sum(far_mask) < min_bucket:
        cont[itm] = _ridge_fit_predict(X_itm, Y_itm, X_itm, ridge_lambda)
        return cont

    pred_itm = np.empty_like(Y_itm)
    pred_itm[near_mask] = _ridge_fit_predict(
        X_itm[near_mask],
        Y_itm[near_mask],
        X_itm[near_mask],
        ridge_lambda,
    )
    pred_itm[far_mask] = _ridge_fit_predict(
        X_itm[far_mask],
        Y_itm[far_mask],
        X_itm[far_mask],
        ridge_lambda,
    )
    cont[itm] = pred_itm
    return cont


# ══════════════════════════════════════════════════════════════════════════
# Barrier MC engine classes
# ══════════════════════════════════════════════════════════════════════════


class _MCBarrierBase(_MCValuationBase):
    """Shared infrastructure for European and American MC barrier valuations."""

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        super().__init__(valuation_ctx)
        self.spec: BarrierSpec = valuation_ctx.spec  # type: ignore[assignment]
        if self.spec.monitoring is BarrierMonitoring.CONTINUOUS and not isinstance(
            self.underlying, GBMProcess
        ):
            raise UnsupportedFeatureError(
                "Continuous barrier monitoring requires GBMProcess (log-normal paths). "
                "Use DISCRETE monitoring for non-GBM processes."
            )

    def _monitoring_indices(self, time_grid: np.ndarray) -> np.ndarray:
        return _resolve_barrier_monitoring_indices(
            time_grid,
            self.spec,
            self.valuation_ctx.pricing_date,
            self.underlying.day_count_convention,
        )


class _MCBarrierEuropeanValuation(_MCBarrierBase):
    """European barrier option valuation using Monte Carlo.

    Supports discrete monitoring (pathwise check) and continuous monitoring
    (Brownian bridge survival-weighting for knock-out, complementary for
    knock-in).  Rebates are handled for AT_HIT and AT_EXPIRY timing.
    """

    def solve(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Simulate paths and compute barrier-adjusted payoffs and rebates.

        Returns
        -------
        tuple of (payoff, weight, rebate_pv, ttm)
            payoff : (num_paths,) undiscounted terminal vanilla payoff
            weight : (num_paths,) barrier weight (0/1 or survival probability)
            rebate_pv : (num_paths,) already-discounted rebate PV
            ttm : float, time-to-maturity in years
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        time_index_end = _resolve_time_index(
            time_grid,
            self.valuation_ctx.maturity,
            "maturity",
            day_count_convention=self.underlying.day_count_convention,
        )

        ttm = self.valuation_ctx._maturity_year_fraction()
        K = self.valuation_ctx.strike
        H = float(self.spec.barrier)
        direction = self.spec.direction
        action = self.spec.action
        sigma = float(self.underlying.volatility)
        n_paths = paths.shape[1]

        payoff = _vanilla_payoff(self.spec.option_type, K, paths[time_index_end])

        t_grid = _year_fractions(
            self.valuation_ctx.pricing_date,
            time_grid,
            day_count_convention=self.underlying.day_count_convention,
        )
        discount_factors = self.valuation_ctx.discount_curve.df(t_grid)

        inception_hit = self.valuation_ctx._barrier_triggered_at_inception()

        monitoring_indices = self._monitoring_indices(time_grid)
        monitoring_indices = monitoring_indices[monitoring_indices <= time_index_end]

        is_continuous = self.spec.monitoring is BarrierMonitoring.CONTINUOUS
        time_deltas = self.underlying._time_deltas()

        if is_continuous:
            weight, rebate_pv = self._continuous_weights(
                paths,
                time_index_end,
                time_deltas,
                t_grid,
                discount_factors,
                H,
                sigma,
                direction,
                action,
                inception_hit,
                n_paths,
            )
        else:
            weight, rebate_pv = self._discrete_weights(
                paths,
                monitoring_indices,
                time_index_end,
                discount_factors,
                H,
                direction,
                action,
                inception_hit,
                n_paths,
            )

        return payoff, weight, rebate_pv, ttm

    def _discrete_weights(
        self,
        paths: np.ndarray,
        monitoring_indices: np.ndarray,
        time_index_end: int,
        discount_factors: np.ndarray,
        H: float,
        direction: BarrierDirection,
        action: BarrierAction,
        inception_hit: bool,
        n_paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Discrete monitoring: pathwise barrier check at monitoring dates."""
        rebate = float(self.spec.rebate)
        rebate_timing = self.spec.rebate_timing

        ever_hit: np.ndarray = np.zeros(n_paths, dtype=bool)
        first_hit_step: np.ndarray = np.full(n_paths, -1, dtype=int)

        if inception_hit:
            ever_hit[:] = True
            first_hit_step[:] = 0

        for idx in monitoring_indices:
            if idx == 0:
                continue
            spots = paths[idx]
            if direction is BarrierDirection.UP:
                step_hit = spots >= H
            else:
                step_hit = spots <= H
            newly_hit = step_hit & ~ever_hit
            first_hit_step[newly_hit] = idx
            ever_hit |= step_hit

        weight: np.ndarray
        if action is BarrierAction.OUT:
            weight = (~ever_hit).astype(float)
        else:
            weight = ever_hit.astype(float)

        rebate_pv = np.zeros(n_paths)
        if rebate > 0.0:
            if action is BarrierAction.OUT:
                knocked_out = ever_hit
                if rebate_timing is RebateTiming.AT_HIT:
                    rebate_pv[knocked_out] = rebate * discount_factors[first_hit_step[knocked_out]]
                else:
                    rebate_pv[knocked_out] = rebate * float(discount_factors[time_index_end])
            else:
                never_in = ~ever_hit
                rebate_pv[never_in] = rebate * float(discount_factors[time_index_end])

        return weight, rebate_pv

    def _continuous_weights(
        self,
        paths: np.ndarray,
        time_index_end: int,
        time_deltas: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
        H: float,
        sigma: float,
        direction: BarrierDirection,
        action: BarrierAction,
        inception_hit: bool,
        n_paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous monitoring: Brownian bridge survival-weighting."""
        rebate = float(self.spec.rebate)
        rebate_timing = self.spec.rebate_timing

        if inception_hit:
            if action is BarrierAction.OUT:
                weight = np.zeros(n_paths)
            else:
                weight = np.ones(n_paths)
            rebate_pv = np.zeros(n_paths)
            if rebate > 0.0:
                if action is BarrierAction.OUT and rebate_timing is RebateTiming.AT_HIT:
                    rebate_pv[:] = rebate
                elif action is BarrierAction.OUT and rebate_timing is RebateTiming.AT_EXPIRY:
                    rebate_pv[:] = rebate * float(discount_factors[time_index_end])
            return weight, rebate_pv

        surv = np.ones(n_paths)
        rebate_pv = np.zeros(n_paths)

        for t in range(1, time_index_end + 1):
            dt_t = float(time_deltas[t - 1])
            p_hit = _brownian_bridge_hit_prob(
                paths[t - 1],
                paths[t],
                H,
                sigma,
                dt_t,
                direction,
            )
            p_surv_step = 1.0 - p_hit

            if (
                rebate > 0.0
                and action is BarrierAction.OUT
                and rebate_timing is RebateTiming.AT_HIT
            ):
                # Approximate the unknown intra-step hit time by discounting to the
                # interval midpoint
                midpoint_df = float(
                    self.valuation_ctx.discount_curve.df(0.5 * (t_grid[t - 1] + t_grid[t]))
                )
                rebate_pv += surv * p_hit * midpoint_df * rebate

            surv *= p_surv_step

        if action is BarrierAction.OUT:
            weight = surv
        else:
            weight = 1.0 - surv

        if rebate > 0.0:
            if action is BarrierAction.OUT and rebate_timing is RebateTiming.AT_EXPIRY:
                rebate_pv = (1.0 - surv) * rebate * float(discount_factors[time_index_end])
            elif action is BarrierAction.IN:
                rebate_pv = surv * rebate * float(discount_factors[time_index_end])

        return weight, rebate_pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return per-path discounted barrier option values."""
        payoff, weight, rebate_pv, ttm = self.solve()
        df = float(self.valuation_ctx.discount_curve.df(ttm))
        return df * payoff * weight + rebate_pv

    def present_value(self) -> float:
        """Calculate PV using Monte Carlo for European barrier option."""
        with log_timing(logger, "MC Barrier European present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Barrier European paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Barrier European",
        )
        return pv


class _MCBarrierAmericanValuation(_MCBarrierBase):
    """American barrier option valuation using Longstaff-Schwartz MC.

    Knock-out: regress on alive+ITM paths; dead paths get zero.
    Knock-in: regress on active+ITM paths only; inactive paths carry
    discounted actual continuation pathwise (no separate regression).
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        super().__init__(valuation_ctx)

        if self.spec.action is not BarrierAction.OUT:
            return

        if self.valuation_ctx._barrier_triggered_at_inception():
            return

        is_down_out_put = (
            self.spec.direction is BarrierDirection.DOWN and self.spec.option_type is OptionType.PUT
        )
        is_up_out_call = (
            self.spec.direction is BarrierDirection.UP and self.spec.option_type is OptionType.CALL
        )
        if is_down_out_put or is_up_out_call:
            warnings.warn(
                "American barrier Monte Carlo LSM can show significant downward bias "
                "for down-and-out puts and up-and-out calls. "
                "Prefer PDE_FD for production pricing.",
                stacklevel=3,
            )

    def _initial_barrier_values(
        self,
        intrinsic: np.ndarray,
        ever_hit: np.ndarray,
    ) -> np.ndarray:
        """Initialise terminal path values before backward induction."""
        values = np.zeros_like(intrinsic)
        rebate = float(self.spec.rebate)
        rebate_timing = self.spec.rebate_timing

        if self.spec.action is BarrierAction.OUT:
            alive_T = ~ever_hit[-1]
            if rebate > 0.0 and rebate_timing is RebateTiming.AT_EXPIRY:
                values[-1] = np.where(alive_T, intrinsic[-1], rebate)
            else:
                values[-1] = intrinsic[-1] * alive_T.astype(float)
            return values

        active_T = ever_hit[-1]
        values[-1] = intrinsic[-1] * active_T.astype(float)
        if rebate > 0.0:
            values[-1] += rebate * (~active_T).astype(float)
        return values

    def _knock_out_step_values(
        self,
        *,
        t: int,
        spot_t: np.ndarray,
        intrinsic_t: np.ndarray,
        next_values: np.ndarray,
        ever_hit_t: np.ndarray,
        first_hit_step: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        """Return one backward-induction step for American knock-out LSM."""
        df_step = discount_factors[t + 1] / discount_factors[t]
        alive = ~ever_hit_t
        itm = (intrinsic_t > 0) & alive

        if self.mc_params.barrier_aware_basis:
            continuation = _ridge_barrier_lsm_continuation(
                S_t=spot_t,
                Y=df_step * next_values,
                itm=itm,
                strike=self.valuation_ctx.strike,
                barrier=float(self.spec.barrier),
                direction=self.spec.direction,
                deg=self.mc_params.deg,
                ridge_lambda=self.mc_params.ridge_lambda,
                min_itm=self.mc_params.min_itm,
            )
        else:
            continuation = _ridge_lsm_continuation(
                S_t=spot_t,
                Y=df_step * next_values,
                itm=itm,
                strike=self.valuation_ctx.strike,
                deg=self.mc_params.deg,
                ridge_lambda=self.mc_params.ridge_lambda,
                min_itm=self.mc_params.min_itm,
            )

        values_t = np.where(
            alive & (intrinsic_t > continuation),
            intrinsic_t,
            df_step * next_values,
        )

        rebate = float(self.spec.rebate)
        rebate_timing = self.spec.rebate_timing
        dead = ~alive
        if rebate > 0.0 and rebate_timing is RebateTiming.AT_HIT:
            just_hit = first_hit_step == t
            values_t = np.where(just_hit, rebate, values_t)
            values_t = np.where(dead & ~just_hit, 0.0, values_t)
        elif rebate > 0.0 and rebate_timing is RebateTiming.AT_EXPIRY:
            rebate_hold_t = rebate * (discount_factors[-1] / discount_factors[t])
            values_t = np.where(dead, rebate_hold_t, values_t)
        else:
            values_t = np.where(dead, 0.0, values_t)
        return values_t

    def _knock_in_step_values(
        self,
        *,
        t: int,
        spot_t: np.ndarray,
        intrinsic_t: np.ndarray,
        next_values: np.ndarray,
        ever_hit_t: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        """Return one backward-induction step for American knock-in LSM."""
        df_step = discount_factors[t + 1] / discount_factors[t]
        active = ever_hit_t
        itm_active = (intrinsic_t > 0) & active

        continuation = _ridge_lsm_continuation(
            S_t=spot_t,
            Y=df_step * next_values,
            itm=itm_active,
            strike=self.valuation_ctx.strike,
            deg=self.mc_params.deg,
            ridge_lambda=self.mc_params.ridge_lambda,
            min_itm=self.mc_params.min_itm,
        )

        return np.where(
            active & (intrinsic_t > continuation),
            intrinsic_t,
            df_step * next_values,
        )

    def solve(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Generate paths, intrinsic payoffs, and barrier state.

        Returns
        -------
        tuple of (spot_paths, intrinsic, ever_hit, first_hit_step,
                  time_index_start, time_index_end)
            spot_paths : (num_exercise_dates, num_paths)
            intrinsic : (num_exercise_dates, num_paths)
            ever_hit : (num_exercise_dates, num_paths) bool — cumulative hit state
            first_hit_step : (num_paths,) int — index within window (-1 if never)
            time_index_start, time_index_end : int
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        time_index_start = _resolve_time_index(
            time_grid,
            self.valuation_ctx.pricing_date,
            "Pricing date",
            day_count_convention=self.underlying.day_count_convention,
        )
        time_index_end = _resolve_time_index(
            time_grid,
            self.valuation_ctx.maturity,
            "maturity",
            day_count_convention=self.underlying.day_count_convention,
        )

        spot_paths = paths[time_index_start : time_index_end + 1]
        n_times, n_paths = spot_paths.shape

        K = self.valuation_ctx.strike
        intrinsic = _vanilla_payoff(self.spec.option_type, K, spot_paths)

        H = float(self.spec.barrier)
        direction = self.spec.direction
        sigma = float(self.underlying.volatility)

        is_continuous = self.spec.monitoring is BarrierMonitoring.CONTINUOUS
        time_deltas = self.underlying._time_deltas()

        assert time_index_start == 0, (
            "time_grid must start at pricing_date; _build_time_grid enforces this"
        )

        monitoring_indices = self._monitoring_indices(time_grid)
        monitoring_indices = monitoring_indices[monitoring_indices <= time_index_end]

        # Build cumulative barrier state
        ever_hit = np.zeros((n_times, n_paths), dtype=bool)
        first_hit_step = np.full(n_paths, -1, dtype=int)

        if self.valuation_ctx._barrier_triggered_at_inception():
            ever_hit[0] = True
            first_hit_step[:] = 0

        # Independent RNG for Bernoulli bridge sampling
        if is_continuous and self.mc_params.random_seed is not None:
            bridge_rng = np.random.default_rng(
                np.random.SeedSequence(self.mc_params.random_seed).spawn(1)[0]
            )
        elif is_continuous:
            bridge_rng = np.random.default_rng()
        else:
            bridge_rng = None

        monitoring_set = set(monitoring_indices.tolist())

        for t in range(1, n_times):
            ever_hit[t] = ever_hit[t - 1]

            if not is_continuous and t not in monitoring_set:
                continue

            spots_prev = spot_paths[t - 1]
            spots_curr = spot_paths[t]

            if is_continuous:
                dt_t = float(time_deltas[t - 1])
                p_hit = _brownian_bridge_hit_prob(
                    spots_prev,
                    spots_curr,
                    H,
                    sigma,
                    dt_t,
                    direction,
                )
                assert bridge_rng is not None
                step_hit = bridge_rng.uniform(size=n_paths) < p_hit
            else:
                if direction is BarrierDirection.UP:
                    step_hit = spots_curr >= H
                else:
                    step_hit = spots_curr <= H

            newly_hit = step_hit & ~ever_hit[t]
            first_hit_step[newly_hit] = t
            ever_hit[t] |= step_hit

        return spot_paths, intrinsic, ever_hit, first_hit_step, time_index_start, time_index_end

    def present_value_pathwise(self) -> np.ndarray:
        """Return per-path discounted barrier option values via LSM backward induction."""
        (
            spot_paths,
            intrinsic,
            ever_hit,
            first_hit_step,
            time_index_start,
            time_index_end,
        ) = self.solve()

        times = self.underlying.time_grid[time_index_start : time_index_end + 1]
        t_grid = _year_fractions(
            self.valuation_ctx.pricing_date,
            times,
            day_count_convention=self.underlying.day_count_convention,
        )
        discount_factors = self.valuation_ctx.discount_curve.df(t_grid)
        n_times, n_paths = spot_paths.shape
        hit_at_inception = bool(ever_hit[0, 0])

        # KO triggered at inception: option is dead, no exercise possible.
        # Mirrors the European short-circuit; rebate (if any) is paid per
        # the rebate_timing rule.  Bypasses the LSM backward induction.
        if self.spec.action is BarrierAction.OUT and hit_at_inception:
            rebate = float(self.spec.rebate)
            if rebate <= 0.0:
                value_0 = 0.0
            elif self.spec.rebate_timing is RebateTiming.AT_HIT:
                value_0 = rebate
            else:
                value_0 = rebate * float(discount_factors[-1])
            return np.full(n_paths, value_0, dtype=float)

        values = self._initial_barrier_values(intrinsic, ever_hit)

        # Backward induction
        for t in range(n_times - 2, 0, -1):
            if self.spec.action is BarrierAction.OUT:
                values[t] = self._knock_out_step_values(
                    t=t,
                    spot_t=spot_paths[t],
                    intrinsic_t=intrinsic[t],
                    next_values=values[t + 1],
                    ever_hit_t=ever_hit[t],
                    first_hit_step=first_hit_step,
                    discount_factors=discount_factors,
                )
            else:
                values[t] = self._knock_in_step_values(
                    t=t,
                    spot_t=spot_paths[t],
                    intrinsic_t=intrinsic[t],
                    next_values=values[t + 1],
                    ever_hit_t=ever_hit[t],
                    discount_factors=discount_factors,
                )

        # t=0: compare expected continuation against immediate exercise.
        # KI is only exercisable if already triggered at inception; otherwise
        # there is no immediate-exercise option (continuation only).
        df0 = discount_factors[1] / discount_factors[0]
        if self.spec.action is BarrierAction.IN and not hit_at_inception:
            immediate_0 = None
        else:
            immediate_0 = float(np.mean(intrinsic[0]))
        return _time_zero_lsm_value(df0 * values[1], immediate_exercise=immediate_0)

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz MC for American barrier option."""
        with log_timing(logger, "MC Barrier American present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Barrier American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
            self.mc_params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Barrier American",
        )
        return pv
