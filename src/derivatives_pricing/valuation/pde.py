"""Finite difference (PDE) valuation implementations.

This module follows the same structure as other valuation modules:
private implementation classes that plug into OptionValuation.

Current scope
-------------
PDE via finite differences for European and American options:
- vanilla call/put and custom payoffs (PayoffSpec)
- time stepping: implicit, explicit, or Crank–Nicolson
- optional Rannacher smoothing for Crank–Nicolson
- spatial grids: spot or log-spot
- American handling: intrinsic projection or Gauss-Seidel/PSOR
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import logging
import math
import datetime as dt
import warnings

import numpy as np

# optional acceleration via numba if available
try:
    from numba import njit as _njit
except ModuleNotFoundError:  # pragma: no cover

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """Identity decorator when numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


from ..enums import DayCountConvention, PDEEarlyExercise, PDEMethod, PDESpaceGrid, OptionType
from ..rates import DiscountCurve
from ..utils import calculate_year_fraction, log_timing
from ..exceptions import (
    ConfigurationError,
    StabilityError,
    UnsupportedFeatureError,
    ValidationError,
)
from .contracts import PayoffBoundaryModel, WingAsymptote
from .params import PDEParams

if TYPE_CHECKING:
    from .core import OptionValuation, UnderlyingData


logger = logging.getLogger(__name__)


def _solve_tridiagonal_thomas(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system Ax = rhs via the Thomas algorithm.

    A has:
      - lower: subdiagonal (length n-1)  -> A[i, i-1]
      - diag:  main diagonal (length n)  -> A[i, i]
      - upper: superdiagonal (length n-1)-> A[i, i+1]

    Parameters
    ----------
    lower
        Subdiagonal coefficients.
    diag
        Main diagonal coefficients.
    upper
        Superdiagonal coefficients.
    rhs
        Right-hand-side vector.

    Returns
    -------
    np.ndarray
        Solution vector of the tridiagonal system.
    """
    n = diag.size
    if rhs.size != n:
        raise ValidationError("rhs length must match diag length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValidationError("lower/upper must have length n-1")

    # Copy to avoid mutating inputs
    a: np.ndarray = lower.astype(float, copy=True)
    d: np.ndarray = diag.astype(float, copy=True)
    c: np.ndarray = upper.astype(float, copy=True)
    y: np.ndarray = rhs.astype(float, copy=True)

    # Forward elimination
    for i in range(1, n):
        w = a[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        y[i] -= w * y[i - 1]

    # Back substitution
    x: np.ndarray = np.empty(n, dtype=float)
    x[-1] = y[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / d[i]
    return x


def _build_tau_grid(
    time_to_maturity: float,
    time_steps: int,
    dividend_taus: list[float],
) -> np.ndarray:
    """Build a tau (time remaining) grid that includes dividend dates."""
    base = np.linspace(0.0, time_to_maturity, time_steps + 1)
    if not dividend_taus:
        return base
    grid = np.unique(np.concatenate([base, np.array(dividend_taus, dtype=float)]))
    grid.sort()
    return grid


def _dividend_tau_schedule(
    *,
    discrete_dividends: Sequence[tuple[dt.datetime, float]],
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    day_count_convention: DayCountConvention,
) -> list[tuple[float, float]]:
    """Return list of (tau, amount) for dividends between pricing_date and maturity.

    The range is closed: ``0.0 <= tau <= ttm``.  Boundary values (tau=0 for
    maturity-date dividends, tau=ttm for pricing-date dividends) are included
    so that ``_vanilla_fd_core`` can apply them as special-case jumps.
    """
    if not discrete_dividends:
        return []

    ttm = calculate_year_fraction(
        pricing_date,
        maturity,
        day_count_convention=day_count_convention,
    )
    schedule: dict[float, float] = {}
    for ex_date, amount in discrete_dividends:
        if pricing_date <= ex_date <= maturity:
            t = calculate_year_fraction(
                pricing_date,
                ex_date,
                day_count_convention=day_count_convention,
            )
            tau = ttm - t
            key = round(float(tau), 12)
            schedule[key] = schedule.get(key, 0.0) + float(amount)
    return sorted(schedule.items())


def _apply_dividend_jump(
    values: np.ndarray,
    grid: np.ndarray,
    amount: float,
    *,
    space_grid: PDESpaceGrid,
) -> None:
    """Apply the cash dividend jump condition V(S,t^-)=V(S-D,t^+)."""
    if amount == 0.0:
        return
    if space_grid is PDESpaceGrid.LOG_SPOT:
        spot_grid = np.exp(grid)
    else:
        spot_grid = grid
    shifted = np.interp(
        spot_grid - amount,
        spot_grid,
        values,
        left=values[0],
        right=values[-1],
    )
    values[:] = shifted


# ---------------------------------------------------------------------------
# Affine boundary-model helpers for custom-payoff boundary conditions
# ---------------------------------------------------------------------------


def _fit_affine_boundary_model(
    payoff_fn: Callable,
    *,
    wing: str,
    spot_samples: np.ndarray,
) -> WingAsymptote:
    """Fit affine boundary model ``payoff(S) ~ slope * S + intercept``.

    Parameters
    ----------
    payoff_fn
        Vectorized payoff callable.
    wing
        ``"left"`` or ``"right"`` boundary label for logging.
    spot_samples
        Spot samples taken directly from the PDE grid near the relevant
        boundary. Using the actual grid nodes makes the fitted affine model
        consistent with the truncated domain the PDE solver uses.

    Returns
    -------
    WingAsymptote
        Fitted slope / intercept pair.
    """
    if wing not in {"left", "right"}:
        raise ConfigurationError(f"wing must be 'left' or 'right', got {wing!r}")

    x = np.asarray(spot_samples, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ConfigurationError("spot_samples must be a 1D array with at least three points")
    if np.any(np.diff(x) <= 0.0):
        raise ConfigurationError("spot_samples must be strictly increasing")

    y = np.asarray(payoff_fn(x), dtype=float)

    # Least-squares fit: y ≈ slope * x + intercept
    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Warn if the affine fit is poor (payoff is genuinely nonlinear on the wing).
    residuals = y - (slope * x + intercept)
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    if ss_tot > 1e-30:
        r_squared = 1.0 - ss_res / ss_tot
        if r_squared < 0.99:
            logger.warning(
                "Affine boundary fit on %s wing has R²=%.4f; boundary values "
                "may be inaccurate. Consider providing explicit PayoffBoundaryModel.",
                wing,
                r_squared,
            )

    return WingAsymptote(slope=float(slope), intercept=float(intercept))


def _continuation_from_affine_boundary_model(
    *,
    spot: float,
    slope: float,
    intercept: float,
    df_tT: float,
    dq_tT: float,
) -> float:
    """Continuation value implied by an affine boundary model.

    If the boundary payoff is approximated by

        payoff(S_T) ~ slope * S_T + intercept

    then under risk-neutral pricing:

        V(S, t) ~ slope * S * dq_tT + intercept * df_tT

    where *dq_tT* is the dividend discount factor and *df_tT* is the risk-free
    discount factor from *t* to *T*.
    """
    return float(slope * spot * dq_tT + intercept * df_tT)


def _boundary_values(
    *,
    option_type: OptionType | None,
    strike: float | None,
    smin: float,
    smax: float,
    df_tT: float,
    dq_tT: float,
    early_exercise: bool,
    payoff_fn: Callable | None = None,
    payoff_boundary_model: PayoffBoundaryModel | None = None,
) -> tuple[float, float]:
    """Dirichlet boundary values for PDE at S=smin (left) and S=smax (right).

    For vanilla call/put, uses standard analytical asymptotics.

    For custom payoffs, uses affine wing boundary models::

        payoff(S) ~ slope * S + intercept
        => V(S, t) ~ slope * S * dq_tT + intercept * df_tT

    For American exercise the boundary is
    ``max(continuation, intrinsic)`` where intrinsic is evaluated directly
    via the payoff callable (not the asymptote).
    """
    # ------------------------------------------------------------------
    # Custom payoff branch
    # ------------------------------------------------------------------
    if payoff_fn is not None:
        if payoff_boundary_model is None:
            raise ConfigurationError(
                "_boundary_values requires a resolved payoff_boundary_model for custom payoffs"
            )

        left_cont = _continuation_from_affine_boundary_model(
            spot=smin,
            slope=payoff_boundary_model.left.slope,
            intercept=payoff_boundary_model.left.intercept,
            df_tT=df_tT,
            dq_tT=dq_tT,
        )
        right_cont = _continuation_from_affine_boundary_model(
            spot=smax,
            slope=payoff_boundary_model.right.slope,
            intercept=payoff_boundary_model.right.intercept,
            df_tT=df_tT,
            dq_tT=dq_tT,
        )

        if early_exercise:
            intrinsic = np.asarray(payoff_fn(np.array([smin, smax], dtype=float)), dtype=float)
            left = max(left_cont, float(intrinsic[0]))
            right = max(right_cont, float(intrinsic[1]))
        else:
            left = left_cont
            right = right_cont

        return float(left), float(right)

    # ------------------------------------------------------------------
    # Vanilla branch
    # ------------------------------------------------------------------
    assert option_type is not None and strike is not None
    if option_type is OptionType.PUT:
        left = strike if early_exercise else strike * df_tT
        right = 0.0
    else:
        left = 0.0
        continuation = smax * dq_tT - strike * df_tT
        intrinsic = smax - strike
        right = max(continuation, intrinsic) if early_exercise else max(continuation, 0.0)
    return float(left), float(right)


def _build_log_grid(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build log-spot grid using Hull's dz heuristic when possible."""
    d_tau = time_to_maturity / time_steps
    dz_hull = volatility * np.sqrt(3.0 * d_tau)

    smax = float(smax_mult * max(spot, strike))
    smin = float(max(max(spot, strike) / smax_mult, 1.0e-8))
    zmin_target = np.log(smin)
    zmax_target = np.log(smax)

    grid_width = spot_steps * dz_hull
    if (zmax_target - zmin_target) > grid_width:
        dz = (zmax_target - zmin_target) / spot_steps
        zmin = zmin_target
        zmax = zmax_target
    else:
        dz = dz_hull
        center = np.log(spot)
        zmin = center - 0.5 * grid_width
        zmax = center + 0.5 * grid_width
        if zmin > zmin_target:
            shift = zmin_target - zmin
            zmin += shift
            zmax += shift
        if zmax < zmax_target:
            shift = zmax_target - zmax
            zmin += shift
            zmax += shift

    Z = np.linspace(zmin, zmax, spot_steps + 1)
    S = np.exp(Z)
    return Z, S, dz


def _spot_operator_coeffs(
    *,
    spot_values: np.ndarray,
    dS: float,
    risk_free_rate: float,
    dividend_rate: float,
    volatility: float,
    hull_discounting: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial operator coefficients on the spot grid.

    When *hull_discounting* is True (Hull's explicit scheme), the rV
    term is excluded from beta and instead applied as an implicit
    divisor ``1 / (1 + r * dt)`` in the time-step function.
    """
    diffusion = (volatility**2) * (spot_values**2) / (dS**2)
    drift = (risk_free_rate - dividend_rate) * spot_values / dS
    gamma = 0.5 * (diffusion - drift)
    beta = -diffusion if hull_discounting else -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _log_operator_coeffs(
    *,
    dz: float,
    risk_free_rate: float,
    dividend_rate: float,
    volatility: float,
    hull_discounting: bool = False,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial operator coefficients on the log-spot grid.

    Returns constant (Toeplitz) arrays of length *size*, matching the
    signature of ``_spot_operator_coeffs`` so callers can treat both
    grids uniformly.

    When *hull_discounting* is True (Hull's explicit scheme), r is
    excluded from beta.
    """
    mu = risk_free_rate - dividend_rate - 0.5 * volatility**2
    diffusion = (volatility**2) / (dz**2)
    drift = mu / dz
    gamma = np.full(size, 0.5 * (diffusion - drift))
    beta = np.full(size, -diffusion if hull_discounting else -(diffusion + risk_free_rate))
    alpha = np.full(size, 0.5 * (diffusion + drift))
    return gamma, beta, alpha


def _scaled_operator_coeffs(
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    d_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = -d_tau * gamma
    b = -d_tau * beta
    c = -d_tau * alpha
    return a, b, c


# ---------------------------------------------------------------------------
# Time-step helpers (extracted from _vanilla_fd_core for readability)
# ---------------------------------------------------------------------------


def _explicit_step(
    V_prev: np.ndarray,
    j: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    intrinsic: np.ndarray | None,
    *,
    r_dt: float = 0.0,
) -> np.ndarray:
    """Explicit (forward-Euler) time step.

    When *r_dt* > 0 (Hull's explicit scheme), the interior update is
    divided by ``(1 + r_dt)`` to apply implicit discounting of the rV
    term.
    """
    V_new = V_prev.copy()
    interior = -a * V_prev[j - 1] + (1.0 - b) * V_prev[j] - c * V_prev[j + 1]
    V_new[j] = interior / (1.0 + r_dt)
    V_new[0] = left
    V_new[-1] = right
    if intrinsic is not None:
        V_new[:] = np.maximum(V_new, intrinsic)
    return V_new


@_njit(cache=True)
def _psor_core(
    x: np.ndarray,
    exercise_j: np.ndarray,
    rhs: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    V_left: float,
    V_right: float,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Numba-accelerated Projected SOR (Gauss-Seidel with overrelaxation)."""
    n = x.shape[0]
    for k in range(n):
        if x[k] < exercise_j[k]:
            x[k] = exercise_j[k]
    iter_used = max_iter
    for iter_idx in range(max_iter):
        max_diff = 0.0
        for k in range(n):
            left_val = x[k - 1] if k > 0 else V_left
            right_val = x[k + 1] if k < n - 1 else V_right
            gs = (rhs[k] - lower[k] * left_val - upper[k] * right_val) / diag[k]
            old = x[k]
            new = old + omega * (gs - old)
            if new < exercise_j[k]:
                new = exercise_j[k]
            x[k] = new
            diff = new - old if new > old else old - new
            if diff > max_diff:
                max_diff = diff
        if max_diff < tol:
            iter_used = iter_idx + 1
            break
    return x, iter_used


def _psor_solve(
    x: np.ndarray,
    exercise_j: np.ndarray,
    rhs: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    V_left: float | np.floating,
    V_right: float | np.floating,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Projected SOR (Gauss-Seidel with overrelaxation) for American exercise.

    Returns the updated interior values and the number of iterations used.
    Delegates to a Numba-JIT compiled inner loop when numba is available.
    """
    return _psor_core(
        x,
        exercise_j,
        rhs,
        lower,
        diag,
        upper,
        float(V_left),
        float(V_right),
        omega,
        tol,
        max_iter,
    )


def _implicit_cn_step(
    V_prev: np.ndarray,
    V: np.ndarray,
    j: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    method: PDEMethod,
    intrinsic: np.ndarray | None,
    american_solver: PDEEarlyExercise,
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
) -> tuple[np.ndarray, int | None]:
    """One implicit or Crank-Nicolson time step (with optional early exercise).

    Returns the updated V array and the PSOR iteration count (None if PSOR was not used).
    """
    if method is PDEMethod.CRANK_NICOLSON:
        a = a * 0.5
        b = b * 0.5
        c = c * 0.5

    # Tridiagonal system (I - θ·dt·L)x = rhs, diagonals are (a, 1+b, c)
    diag = 1.0 + b

    if method is PDEMethod.IMPLICIT:
        rhs = V_prev[j].copy()
    else:
        rhs = -a * V_prev[j - 1] + (1.0 - b) * V_prev[j] - c * V_prev[j + 1]

    V[0] = left
    V[-1] = right

    rhs_adj = rhs.copy()
    rhs_adj[0] -= a[0] * V[0]
    rhs_adj[-1] -= c[-1] * V[-1]

    x = _solve_tridiagonal_thomas(a[1:], diag, c[:-1], rhs_adj)
    psor_iters: int | None = None

    if intrinsic is None:
        V[j] = x
    else:
        exercise_j = intrinsic[j]
        if american_solver is PDEEarlyExercise.INTRINSIC:
            V[j] = np.maximum(x, exercise_j)
        else:
            x, psor_iters = _psor_solve(
                x,
                exercise_j,
                rhs,
                a,
                diag,
                c,
                float(V[0]),
                float(V[-1]),
                float(omega),
                float(tol),
                int(max_iter),
            )
            V[j] = x

    return V, psor_iters


def _validate_fd_inputs(
    *,
    option_type: OptionType | None,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    volatility: float,
    discount_curve: DiscountCurve,
    early_exercise: bool,
    method: PDEMethod,
    american_solver: PDEEarlyExercise,
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
    payoff_fn: Callable | None = None,
) -> None:
    """Validate FD PDE inputs before grid construction."""
    if payoff_fn is None and option_type not in (OptionType.CALL, OptionType.PUT):
        raise UnsupportedFeatureError("FD PDE valuation supports only vanilla CALL/PUT.")
    if time_to_maturity <= 0:
        raise ValidationError("time_to_maturity must be positive")
    if spot_steps < 3:
        raise ValidationError("spot_steps must be >= 3")
    if time_steps < 1:
        raise ValidationError("time_steps must be >= 1")
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if discount_curve is None:
        raise ValidationError("discount_curve is required for PDE valuation")
    if early_exercise and american_solver is PDEEarlyExercise.GAUSS_SEIDEL:
        if omega is None or tol is None or max_iter is None:
            raise ValidationError(
                "PSOR params (omega/tol/max_iter) are required for early exercise"
            )
    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and (
        american_solver is PDEEarlyExercise.GAUSS_SEIDEL
    ):
        raise UnsupportedFeatureError("GAUSS_SEIDEL is not supported with explicit time stepping")


def _check_explicit_spot_stability(
    *,
    tau_grid: np.ndarray,
    volatility: float,
    smax: float,
    dS: float,
    time_to_maturity: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    hull_discounting: bool,
) -> None:
    r"""CFL-style stability checks for an explicit scheme on a uniform spot grid.

    The dominant restriction comes from the diffusion term::

        dt <= dS² / (σ² S_max²)

    When *hull_discounting* is ``False`` (pure explicit), the reaction
    term :math:`-rV` is discretised explicitly too, giving the tighter
    bound::

        dt <= 1 / (σ² S_max² / dS² + r_max)

    With Hull implicit discounting, ``r`` is handled implicitly via the
    :math:`1/(1 + r \Delta t)` divisor so only diffusion matters.

    A secondary check enforces the central-difference stencil monotonicity
    condition :math:`\alpha, \gamma \ge 0`, i.e. diffusion dominates drift
    at the worst-case grid node (``S_max``).

    Raises ``StabilityError`` if any condition is violated.
    """
    if tau_grid.size < 2:
        return

    dt_steps = np.diff(tau_grid).astype(float)
    max_dt = float(np.max(dt_steps))
    if max_dt <= 0.0:
        return

    # Forward rates for each tau step (calendar times: t = T - tau)
    t_prev = time_to_maturity - tau_grid[:-1]
    t_next = time_to_maturity - tau_grid[1:]
    r_steps = np.array(
        [discount_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
        dtype=float,
    )
    if dividend_curve is not None:
        q_steps = np.array(
            [dividend_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
            dtype=float,
        )
    else:
        q_steps = np.zeros_like(r_steps)

    rq_abs_max = float(np.max(np.abs(r_steps - q_steps)))

    # (A) Diffusion CFL bound
    diffusion_max = (volatility**2) * (smax**2) / (dS**2)

    if hull_discounting:
        # r handled implicitly — only diffusion constrains dt
        dt_max = 1.0 / diffusion_max
        mode = "Hull implicit discounting"
    else:
        # Pure explicit: tighten for the reaction term -rV
        r_pos = max(float(np.max(r_steps)), 0.0)
        dt_max = (
            1.0 / (diffusion_max + r_pos) if (diffusion_max + r_pos) > 0.0 else 1.0 / diffusion_max
        )
        mode = "pure explicit discounting"

    if max_dt > dt_max:
        min_steps = int(math.ceil(time_to_maturity / dt_max))
        raise StabilityError(
            f"Explicit spot-grid scheme likely unstable (CFL violation, {mode}). "
            f"max_dt={max_dt:.4g} exceeds dt_max={dt_max:.4g}. "
            f"Increase time_steps to >= {min_steps}, or use log-spot/implicit/CN."
        )

    # (B) Drift monotonicity: central-difference stencil requires
    #     diffusion >= |drift| at worst-case S for alpha, gamma >= 0.
    drift_max = rq_abs_max * smax / dS
    if drift_max > diffusion_max:
        raise StabilityError(
            "Explicit spot-grid scheme likely unstable/oscillatory: drift dominates "
            "diffusion in the central-difference stencil. "
            f"diffusion_max={diffusion_max:.4g}, drift_max={drift_max:.4g}. "
            "Refine the spot grid (smaller dS), reduce smax, or use log-spot/implicit/CN."
        )


def _build_time_step_schedule(
    tau_grid: np.ndarray,
    method: PDEMethod,
    rannacher_steps: int,
) -> list[tuple[float, float, PDEMethod]]:
    """Build the time-step schedule from the tau grid.

    For Crank-Nicolson with Rannacher smoothing (Pooley-Vetzal-Forsyth 2003),
    the first *rannacher_steps* intervals are each replaced by two implicit
    (backward Euler) half-steps of size d_tau/2.  This damps payoff
    non-smoothness while preserving the overall time-grid structure.
    For all other methods the schedule is a straightforward pass-through.
    """
    steps: list[tuple[float, float, PDEMethod]] = []
    for n in range(1, tau_grid.size):
        tau_start = float(tau_grid[n - 1])
        tau_end = float(tau_grid[n])
        if method is PDEMethod.CRANK_NICOLSON and rannacher_steps > 0 and n <= rannacher_steps:
            tau_mid = 0.5 * (tau_start + tau_end)
            steps.append((tau_start, tau_mid, PDEMethod.IMPLICIT))
            steps.append((tau_mid, tau_end, PDEMethod.IMPLICIT))
        else:
            steps.append((tau_start, tau_end, method))
    return steps


def _vanilla_fd_core(
    *,
    spot: float,
    strike: float | None,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType | None,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    payoff_fn: Callable | None = None,
    payoff_boundary_model: PayoffBoundaryModel | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Core finite-difference solver for option valuation.

    Supports vanilla CALL/PUT (via *option_type* and *strike*) and
    arbitrary payoffs (via *payoff_fn*).  When *payoff_fn* is provided
    it takes precedence and *option_type*/*strike* may be ``None``.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_final, V_prev, last_dtau)``
        where *V_prev* is the value slice one time step before pricing
        time and *last_dtau* is the size of that step (both needed for
        theta extraction from the grid).
    """
    _validate_fd_inputs(
        option_type=option_type,
        time_to_maturity=time_to_maturity,
        spot_steps=spot_steps,
        time_steps=time_steps,
        volatility=volatility,
        discount_curve=discount_curve,
        early_exercise=early_exercise,
        method=method,
        american_solver=american_solver,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
        payoff_fn=payoff_fn,
    )

    # For grid sizing, use strike when available, otherwise use spot.
    # After the grid is built, normalize smin/smax to the ACTUAL grid
    # boundaries so all downstream code uses a single consistent meaning.
    ref_price = max(spot, strike) if strike is not None else spot
    smax = float(smax_mult * ref_price)
    if space_grid is PDESpaceGrid.SPOT:
        grid = np.linspace(0.0, smax, spot_steps + 1)
        S = grid
        dS = smax / spot_steps
    else:
        grid, S, dz = _build_log_grid(
            spot=spot,
            strike=strike if strike is not None else spot,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
        )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices 1..M-1

    # Terminal payoff at maturity
    if payoff_fn is not None:
        payoff = np.asarray(payoff_fn(S), dtype=float)
    elif option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    # Resolve affine wing boundary models once (used for boundary
    # conditions on every time step). Prefer explicit metadata; fall back
    # to a local affine fit on the actual PDE boundary nodes.
    if payoff_fn is not None:
        if payoff_boundary_model is None:
            payoff_boundary_model = PayoffBoundaryModel(
                left=_fit_affine_boundary_model(payoff_fn, wing="left", spot_samples=S[:4]),
                right=_fit_affine_boundary_model(payoff_fn, wing="right", spot_samples=S[-4:]),
            )
        elif space_grid is PDESpaceGrid.LOG_SPOT:
            warnings.warn(
                "Explicit PayoffBoundaryModel with LOG_SPOT grid is interpreted as an affine "
                "boundary model on the finite truncated PDE domain, not as a true payoff tail "
                "asymptote. Ensure the supplied boundary model is appropriate at the actual grid "
                "boundaries.",
                stacklevel=2,
            )

    V = payoff.copy()  # V at tau=0 (maturity)
    intrinsic = payoff if early_exercise else None

    schedule = dividend_schedule or []
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}

    # Maturity-date dividend (tau=0): apply as an immediate jump
    # V(S,T⁻)=V(S-D,T⁺) right after setting the terminal condition.
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V[:] = np.maximum(V, payoff)

    # Pricing-date dividend (tau=ttm): will be applied as a spot shift
    # at interpolation time.  Remove from the map so it's not applied
    # as a mid-grid jump during time-stepping.
    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    dividend_taus = list(dividend_map.keys())
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, dividend_taus)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and space_grid is PDESpaceGrid.SPOT:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=dS,
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            hull_discounting=method is PDEMethod.EXPLICIT_HULL,
        )

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    df_0T = float(discount_curve.df(time_to_maturity))  # P(0,T)
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))  # Dq(0,T)
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)

    V_prev = V.copy()
    last_dtau = 0.0

    for tau_prev, tau_curr, method_used in steps:
        d_tau = tau_curr - tau_prev
        t_prev = time_to_maturity - tau_prev
        t_curr = time_to_maturity - tau_curr

        r = float(discount_curve.forward_rate(t_curr, t_prev))
        if dividend_curve is not None:
            q = float(dividend_curve.forward_rate(t_curr, t_prev))
        else:
            q = 0.0

        hull_discounting = method_used is PDEMethod.EXPLICIT_HULL

        if space_grid is PDESpaceGrid.SPOT:
            gamma, beta, alpha = _spot_operator_coeffs(
                spot_values=S[1:-1],
                dS=dS,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                hull_discounting=hull_discounting,
            )
        else:
            gamma, beta, alpha = _log_operator_coeffs(
                dz=dz,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                hull_discounting=hull_discounting,
                size=spot_steps - 1,
            )

        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smin=smin,
            smax=smax,
            df_tT=df_tT,
            dq_tT=dq_tT,
            early_exercise=early_exercise,
            payoff_fn=payoff_fn,
            payoff_boundary_model=payoff_boundary_model,
        )

        V_prev = V.copy()
        last_dtau = d_tau

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        intrinsic_for_step = intrinsic if early_exercise else None

        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V = _explicit_step(
                V_prev,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic_for_step,
                r_dt=r * d_tau if hull_discounting else 0.0,
            )
        else:
            V, psor_iters = _implicit_cn_step(
                V_prev,
                V,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                intrinsic_for_step,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):
                    psor_not_converged += 1

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    # Apply pricing-date dividend: the input spot is cum-dividend, so
    # interpolate at S₀ − D to get the ex-dividend option value.
    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = np.interp(interp_spot, S, V)
    return price, S, V, V_prev, last_dtau


class _FDGridGreeksMixin:
    """Mixin providing delta/gamma/theta extracted from the PDE solution grid.

    Subclasses must define ``_solve()`` returning
    ``(price, S, V, V_prev, last_dtau)``.
    """

    valuation_ctx: OptionValuation
    underlying: UnderlyingData

    def _solve(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]: ...

    def _grid_greeks_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Run the PDE solve and locate the spot node.

        Returns
        -------
        S, V, V_prev, last_dtau, j
            The spot grid, value vector, previous-step value vector,
            last time-step size, and the spot-grid index closest to
            the current spot.
        """
        _, S, V, V_prev, last_dtau = self._solve()
        spot = float(self.underlying.initial_value)
        j = int(np.searchsorted(S, spot))
        # Clamp to interior so the 3-point stencil is valid.
        j = max(1, min(j, len(S) - 2))
        return S, V, V_prev, last_dtau, j

    def delta(self) -> float:
        r"""Grid delta via central differences at the spot node.

        .. math::

            \Delta \approx \frac{V_{j+1} - V_{j-1}}{S_{j+1} - S_{j-1}}
        """
        S, V, _, _, j = self._grid_greeks_data()
        return float((V[j + 1] - V[j - 1]) / (S[j + 1] - S[j - 1]))

    def gamma(self) -> float:
        r"""Grid gamma via the standard second-order stencil.

        .. math::

            \Gamma \approx \frac{V_{j+1} - 2V_j + V_{j-1}}
                               {\tfrac12(S_{j+1} - S_{j-1}) \cdot (S_{j+1} - S_{j-1})/2}

        For a uniform grid this reduces to
        :math:`(V_{j+1} - 2V_j + V_{j-1}) / h^2`.
        """
        S, V, _, _, j = self._grid_greeks_data()
        h_up = S[j + 1] - S[j]
        h_dn = S[j] - S[j - 1]
        return float(
            2.0
            * (V[j + 1] * h_dn + V[j - 1] * h_up - V[j] * (h_up + h_dn))
            / (h_up * h_dn * (h_up + h_dn))
        )

    def theta(self) -> float:
        r"""Grid theta via backward difference between the last two time
        slices.

        .. math::

            \Theta \approx \frac{V^{n}(S_0) - V^{n-1}(S_0)}{\Delta t}

        Returned per **calendar day** (divided by 365).
        """
        S, V, V_prev, last_dtau, j = self._grid_greeks_data()
        if last_dtau <= 0.0:
            return 0.0
        theta_annual = (V_prev[j] - V[j]) / last_dtau
        return float(theta_annual / 365.0)


class _FDValuationBase(_FDGridGreeksMixin):
    """Base class for European/American FD valuation."""

    _early_exercise: bool = False

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying  # type: ignore[assignment]

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
        pv, S, V, *_ = self._solve()
        return pv, S, V

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the PDE finite-difference solve."""
        params = self.valuation_ctx.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")

        if self._early_exercise:
            logger.debug(
                "PDE American method=%s grid=%s solver=%s spot_steps=%d time_steps=%d",
                params.method.value,
                params.space_grid.value,
                params.american_solver.value,
                params.spot_steps,
                params.time_steps,
            )
        else:
            logger.debug(
                "PDE European method=%s grid=%s spot_steps=%d time_steps=%d",
                params.method.value,
                params.space_grid.value,
                params.spot_steps,
                params.time_steps,
            )

        spot = float(self.underlying.initial_value)
        strike = self.valuation_ctx.strike
        volatility = float(self.underlying.volatility)
        discount_curve = self.valuation_ctx.discount_curve
        dividend_curve = self.underlying.dividend_curve
        discrete_dividends = self.underlying.discrete_dividends

        time_to_maturity = calculate_year_fraction(
            self.valuation_ctx.pricing_date,
            self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=discrete_dividends,
            pricing_date=self.valuation_ctx.pricing_date,
            maturity=self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )

        smax_mult = float(params.smax_mult)
        spot_steps = int(params.spot_steps)
        time_steps = int(params.time_steps)

        # Custom payoff support: extract payoff callable and boundary model from PayoffSpec
        spec = self.valuation_ctx.spec
        custom_payoff = getattr(spec, "payoff", None) if strike is None else None
        custom_boundary_model = getattr(spec, "boundary_model", None) if strike is None else None

        return _vanilla_fd_core(
            spot=spot,
            strike=float(strike) if strike is not None else None,
            time_to_maturity=float(time_to_maturity),
            volatility=volatility,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=self.valuation_ctx.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            early_exercise=self._early_exercise,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
            american_solver=params.american_solver
            if self._early_exercise
            else PDEEarlyExercise.INTRINSIC,
            omega=float(params.omega) if self._early_exercise else None,
            tol=float(params.tol) if self._early_exercise else None,
            max_iter=int(params.max_iter) if self._early_exercise else None,
            payoff_fn=custom_payoff,
            payoff_boundary_model=custom_boundary_model,
        )

    def present_value(self) -> float:
        """Return present value from the PDE solve."""
        params = self.valuation_ctx.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")
        label = "PDE American" if self._early_exercise else "PDE European"
        with log_timing(logger, f"{label} present_value", params.log_timings):
            pv, *_ = self._solve()
        return float(pv)


class _FDEuropeanValuation(_FDValuationBase):
    """European option valuation using PDE finite differences."""

    _early_exercise = False


class _FDAmericanValuation(_FDValuationBase):
    """American option valuation using PDE finite differences."""

    _early_exercise = True
