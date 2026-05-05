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

Discrete barriers
-----------------
Uses Boyle-Tian-inspired half-step barrier placement for discrete
monitoring.  This tends to improve accuracy when the monitoring interval
is not too fine relative to the PDE grid time-step.  If monitoring is
extremely dense relative to the PDE grid, the correction can over-shift;
in that regime the contract is close to continuously monitored and
``BarrierMonitoring.CONTINUOUS`` is usually the better model choice.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import logging
import math
import threading
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


from ..enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    OptionType,
    RebateTiming,
)
from ..rates import DiscountCurve
from ..utils import calculate_year_fraction, log_timing
from ..exceptions import (
    ConfigurationError,
    StabilityError,
    UnsupportedFeatureError,
    ValidationError,
)
from .contracts import BarrierSpec, PayoffSpec, PayoffBoundaryModel, VanillaSpec, WingBoundary
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
    extra_taus: list[float],
) -> np.ndarray:
    """Build a tau grid that snaps to dividend and monitoring dates."""
    base = np.linspace(0.0, time_to_maturity, time_steps + 1)
    if not extra_taus:
        return base
    # Round to 12dp on both inputs so np.unique can collapse near-duplicates.
    # Extras already arrive at 12dp from the upstream tau converters
    # (_barrier_monitoring_taus, _dividend_tau_schedule); the linspace base
    # is bit-fresh.  Without the round, an extra that lands on the same
    # physical time as a base point survives as a sub-1e-12 neighbour, then
    # `T - tau` collapses both to the same float and downstream forward-rate
    # calls raise on dt == 0.
    grid = np.unique(np.round(np.concatenate([base, np.array(extra_taus, dtype=float)]), 12))
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
    so that ``_fd_core`` can apply them as special-case jumps.
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
) -> WingBoundary:
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
    WingBoundary
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

    return WingBoundary(slope=float(slope), intercept=float(intercept))


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

    For vanilla call/put, uses standard analytical asymptotic boundary
    conditions.

    For custom payoffs, uses affine wing boundary models::

        payoff(S) ~ slope * S + intercept
        => V(S, t) ~ slope * S * dq_tT + intercept * df_tT

    For custom payoffs, the affine boundary wings are interpreted at the
    actual finite grid boundaries ``smin`` and ``smax`` of the truncated PDE
    domain rather than as true infinite-domain asymptotics.

    For American exercise the boundary is
    ``max(continuation, intrinsic)`` where intrinsic is evaluated directly
    via the payoff callable (not the boundary model).
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
        intrinsic = max(strike - smin, 0.0)
        continuation = strike * df_tT - smin * dq_tT
        left = max(continuation, intrinsic) if early_exercise else continuation
        right = 0.0
    else:
        left = 0.0
        continuation = smax * dq_tT - strike * df_tT
        intrinsic = max(smax - strike, 0.0)
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
    method: PDEMethod,
    anchor_spot: float | None = None,
    anchor_half_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build log-spot grid.

    For the explicit-family schemes (``EXPLICIT``, ``EXPLICIT_HULL``) the
    grid construction preserves Hull's heuristic scale
    ``dz_hull = vol * sqrt(3 * dt)`` when the target log-domain fits within
    ``spot_steps * dz_hull``. For ``EXPLICIT_HULL`` this is the special
    spacing that recovers the trinomial-equivalent explicit discretization
    with up/mid/down probabilities ``1/6, 2/3, 1/6``.

    For unconditionally stable schemes (``IMPLICIT``, ``CRANK_NICOLSON``)
    ``spot_steps`` controls the spatial density directly:
    ``dz = (zmax_target - zmin_target) / spot_steps``.

    When ``anchor_spot`` is provided, the grid is sized so that the anchor
    lies exactly on an interior node by default. If ``anchor_half_step`` is
    true, the anchor instead lies halfway between two adjacent nodes. In both
    cases the resulting domain is a (possibly slight) superset of
    ``[zmin_target, zmax_target]``. For CN/IMPLICIT this is achieved by
    recomputing ``dz`` from the binding half (left or right of the anchor),
    i.e. the side that requires the larger uniform ``dz`` to keep the anchor
    on-node or at a cell midpoint while still covering the target domain. The
    other side can then have up to roughly one cell of slack. For explicit
    schemes ``dz`` is fixed by Hull's stability heuristic, so the grid is
    shifted in place while keeping strict cover of the target domain.
    """
    if anchor_half_step and anchor_spot is None:
        raise ValidationError("anchor_half_step requires anchor_spot to be provided")

    smax = float(smax_mult * max(spot, strike))
    smin = float(max(max(spot, strike) / smax_mult, 1.0e-8))
    zmin_target = np.log(smin)
    zmax_target = np.log(smax)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        d_tau = time_to_maturity / time_steps
        dz_hull = volatility * np.sqrt(3.0 * d_tau)
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
    else:
        # Unconditionally stable schemes: honor spot_steps directly.
        dz = (zmax_target - zmin_target) / spot_steps
        zmin = zmin_target
        zmax = zmax_target

    if anchor_spot is None:
        Z = zmin + dz * np.arange(spot_steps + 1, dtype=float)
        return Z, np.exp(Z), dz

    if anchor_spot <= 0.0:
        raise ValidationError("anchor_spot must be positive for log-spot grids")

    z_anchor = float(np.log(anchor_spot))
    if not (zmin_target <= z_anchor <= zmax_target):
        raise ValidationError("anchor_spot must lie within the log-grid target domain")

    anchor_offset = 0.5 if anchor_half_step else 0.0

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        # Explicit schemes use Hull's dz_hull heuristic, which leaves
        # ``grid_width = spot_steps * dz`` strictly larger than the
        # target span (when not capped). dz is fixed by stability, so
        # we shift the grid in place while keeping strict cover of
        # ``[zmin_target, zmax_target]``. If the target span is already
        # capped exactly by ``spot_steps * dz``, exact anchoring is only
        # possible when the anchor happens to lie on that fixed grid.
        j_min = max(
            0,
            int(math.ceil((z_anchor - zmin_target) / dz - anchor_offset - 1.0e-12)),
        )
        j_max = min(
            spot_steps - 1 if anchor_half_step else spot_steps,
            int(math.floor(spot_steps - (zmax_target - z_anchor) / dz - anchor_offset + 1.0e-12)),
        )
        if j_min > j_max:
            raise StabilityError("Unable to align anchor_spot on the log grid with current setup")
        preferred_index = int(round((z_anchor - zmin) / dz - anchor_offset))
        j_anchor = min(max(preferred_index, j_min), j_max)
    else:
        # CN/IMPLICIT: dz is free, so instead of shifting a fixed-dz
        # grid (which forces an unsatisfiable strict-cover constraint
        # when dz exactly tiles the target span), we *grow* dz on the
        # binding half. Pick the integer node closest to where the
        # anchor naturally falls, then compute the dz required to cover
        # the left and right halves separately; whichever side requires
        # the larger dz is the binding side, and the other side absorbs
        # the slack. The result is a uniform grid that:
        #   - places the anchor exactly on an interior node,
        #   - is strictly tight to the target on the binding side,
        #   - has up to one cell of slack outside the target on the
        #     other side (i.e. a slight superset of the target — never
        #     under-covers),
        #   - costs at most ~1/(spot_steps - 1) extra dz vs the bare-
        #     minimum tile of the target span.
        span = zmax_target - zmin_target
        j_opt = int(round(spot_steps * (z_anchor - zmin_target) / span - anchor_offset))
        j_anchor = max(0 if anchor_half_step else 1, min(spot_steps - 1, j_opt))
        left_dz = (z_anchor - zmin_target) / (j_anchor + anchor_offset)
        right_dz = (zmax_target - z_anchor) / (spot_steps - j_anchor - anchor_offset)
        dz = max(left_dz, right_dz)

    zmin = z_anchor - (j_anchor + anchor_offset) * dz
    Z = zmin + dz * np.arange(spot_steps + 1, dtype=float)
    if not anchor_half_step:
        Z[j_anchor] = z_anchor
    return Z, np.exp(Z), dz


def _build_spot_grid(
    *,
    smin: float,
    smax: float,
    spot_steps: int,
    anchor_spot: float | None = None,
    anchor_half_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a uniform spot grid, optionally aligning an anchor to a node or midpoint."""
    if anchor_spot is None:
        grid = np.linspace(smin, smax, spot_steps + 1)
        dS = (smax - smin) / spot_steps
        return grid, grid, dS

    if not (smin < anchor_spot < smax):
        raise ValidationError("anchor_spot must lie strictly inside the spot-grid domain")

    ratio = (anchor_spot - smin) / (smax - smin)
    if anchor_half_step:
        j_max = min(spot_steps - 1, int(math.floor(spot_steps * ratio - 0.5 + 1.0e-12)))
        if j_max < 0:
            raise StabilityError("Unable to align anchor_spot on the spot grid with current setup")
        preferred_index = int(round(spot_steps * ratio - 0.5))
        j_anchor = min(max(preferred_index, 0), j_max)
        dS = (anchor_spot - smin) / (j_anchor + 0.5)
        grid = smin + dS * np.arange(spot_steps + 1, dtype=float)
        return grid, grid, dS

    j_max = min(spot_steps - 1, int(math.floor(spot_steps * ratio + 1.0e-12)))
    if j_max < 1:
        raise StabilityError("Unable to align anchor_spot on the spot grid with current setup")

    preferred_index = int(round(spot_steps * ratio))
    j_anchor = min(max(preferred_index, 1), j_max)
    dS = (anchor_spot - smin) / j_anchor
    grid = smin + dS * np.arange(spot_steps + 1, dtype=float)
    grid[j_anchor] = anchor_spot
    return grid, grid, dS


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
# Time-step helpers (extracted from _fd_core for readability)
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
    american_solver: PDEEarlyExercise | None,
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
    american_solver: PDEEarlyExercise | None,
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
    if early_exercise and american_solver is None:
        raise ValidationError("american_solver is required when early_exercise=True")
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


def _fd_core(
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
    american_solver: PDEEarlyExercise | None = None,
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
            method=method,
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
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
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

    @staticmethod
    def _spot_grid_index(S: np.ndarray, spot: float) -> int:
        """Return the nearest interior grid index to the current spot."""
        j = int(np.searchsorted(S, spot))
        return max(1, min(j, len(S) - 2))

    @staticmethod
    def _grid_gamma_at_index(S: np.ndarray, V: np.ndarray, j: int) -> float:
        """Return the non-uniform three-point gamma stencil at index ``j``."""
        h_up = S[j + 1] - S[j]
        h_dn = S[j] - S[j - 1]
        return float(
            2.0
            * (V[j + 1] * h_dn + V[j - 1] * h_up - V[j] * (h_up + h_dn))
            / (h_up * h_dn * (h_up + h_dn))
        )

    @staticmethod
    def _grid_delta_at_spot(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Parabolic-Lagrange first derivative evaluated exactly at ``spot``.

        Differentiates the quadratic interpolant through nodes
        ``(S[j-1], S[j], S[j+1])`` at the actual spot rather than at the
        nearest node ``S[j]``. This removes an O(spot - S[j]) discretization
        bias that shows up whenever the query spot doesn't coincide with a
        grid node.

        This is especially important for KI parity as the vanilla and KO
        surfaces live on different grids.
        """
        x0, x1, x2 = S[j - 1], S[j], S[j + 1]
        v0, v1, v2 = V[j - 1], V[j], V[j + 1]
        return float(
            v0 * (2.0 * spot - x1 - x2) / ((x0 - x1) * (x0 - x2))
            + v1 * (2.0 * spot - x0 - x2) / ((x1 - x0) * (x1 - x2))
            + v2 * (2.0 * spot - x0 - x1) / ((x2 - x0) * (x2 - x1))
        )

    @staticmethod
    def _grid_gamma_at_spot(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Cubic-Lagrange second derivative evaluated exactly at ``spot``.

        A 3-point parabolic fit gives a *constant* second derivative
        (``f''(x) = 2a``), so a parabolic at-spot evaluation is identical
        to the at-index value. A 4-point cubic Lagrange yields
        ``f''(x) = 6ax + 2b`` (linear in ``x``), so evaluating at exactly
        ``spot`` decouples the result from the local node placement and
        removes an O(spot - S[j]) bias whenever the query spot doesn't
        coincide with a grid node.

        Especially important for KI parity: vanilla and KO live on
        different grids, so both gammas need to be referenced to the same
        physical ``spot`` rather than to differently-placed ``S[j]`` nodes.

        Uses nodes ``(S[j-1], S[j], S[j+1], S[j+2])``; the index ``j`` is
        clamped so all four neighbours exist.
        """
        n = len(S)
        jc = max(1, min(j, n - 3))
        x0, x1, x2, x3 = S[jc - 1], S[jc], S[jc + 1], S[jc + 2]
        v0, v1, v2, v3 = V[jc - 1], V[jc], V[jc + 1], V[jc + 2]
        # For the cubic Lagrange basis L_i(x) = prod_{k!=i}(x - x_k) / D_i,
        # L_i''(x) = (6x - 2 * sum_{k!=i} x_k) / D_i.
        s0 = x1 + x2 + x3
        s1 = x0 + x2 + x3
        s2 = x0 + x1 + x3
        s3 = x0 + x1 + x2
        d0 = (x0 - x1) * (x0 - x2) * (x0 - x3)
        d1 = (x1 - x0) * (x1 - x2) * (x1 - x3)
        d2 = (x2 - x0) * (x2 - x1) * (x2 - x3)
        d3 = (x3 - x0) * (x3 - x1) * (x3 - x2)
        return float(
            v0 * (6.0 * spot - 2.0 * s0) / d0
            + v1 * (6.0 * spot - 2.0 * s1) / d1
            + v2 * (6.0 * spot - 2.0 * s2) / d2
            + v3 * (6.0 * spot - 2.0 * s3) / d3
        )

    # Relative tolerance for the cubic-vs-parabolic gamma agreement check;
    # if cubic disagrees with parabolic by more than this fraction (of the
    # larger magnitude), we treat the cubic as polluted by PDE noise and
    # fall back to the parabolic value.
    _GAMMA_CUBIC_PARABOLIC_REL_TOL: float = 0.5

    @staticmethod
    def _grid_gamma_safe(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Cubic-at-spot gamma with parabolic-at-index fallback.

        The cubic 4-point stencil is more accurate where ``V`` is locally
        smooth, but it has wider reach and larger basis-function weights
        than the parabolic 3-point stencil, so it amplifies PDE noise more
        aggressively (e.g. PSOR oscillations near American exercise
        boundaries, or noise leaking back from the KI coupling step in
        the two-surface solver). When the two stencils disagree by more
        than ``_GAMMA_CUBIC_PARABOLIC_REL_TOL``, we treat the cubic as
        polluted and fall back to the parabolic value.

        For the well-behaved scenarios cubic and parabolic agree closely
        and the cubic is returned (slightly more accurate); for the
        pathological scenarios the parabolic safety net catches the
        amplified noise.
        """
        parabolic = _FDGridGreeksMixin._grid_gamma_at_index(S, V, j)
        cubic = _FDGridGreeksMixin._grid_gamma_at_spot(S, V, j, spot)
        scale = max(abs(parabolic), abs(cubic), 1e-6)
        if abs(cubic - parabolic) > _FDGridGreeksMixin._GAMMA_CUBIC_PARABOLIC_REL_TOL * scale:
            return parabolic
        return cubic

    def _solve(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]: ...

    def _grid_greeks_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
        """Run the PDE solve and locate the spot node.

        Returns
        -------
        S, V, V_prev, last_dtau, j, spot
            The spot grid, value vector, previous-step value vector,
            last time-step size, the spot-grid index closest to the
            current spot, and the spot itself.
        """
        _, S, V, V_prev, last_dtau = self._solve()
        spot = float(self.underlying.initial_value)
        j = self._spot_grid_index(S, spot)
        return S, V, V_prev, last_dtau, j, spot

    def _intrinsic_short_circuit_greeks(
        self, S: np.ndarray, V: np.ndarray, j: int
    ) -> tuple[float, float] | None:
        """Return ``(delta, gamma)`` if the spot node sits in the
        early-exercise region of an American option, else ``None``.

        When ``V[j]`` equals the intrinsic value at ``S[j]``, the node
        lies in the early-exercise region and the local value function is
        ``V(s) = max(K - s, 0)`` (put) or ``V(s) = max(s - K, 0)`` (call).
        Local greeks are then exact (``delta = ±1``, ``gamma = 0``) and
        the PDE stencil extraction is unreliable due to PSOR oscillations
        near the exercise boundary — both parabolic and cubic stencils
        can produce wildly wrong values because the noise is in the
        underlying ``V`` samples, not in the stencil.

        Only fires for American options on a CALL/PUT spec with strictly
        positive intrinsic value at ``S[j]``.
        """
        spec = self.valuation_ctx.spec
        if not isinstance(spec, (VanillaSpec, BarrierSpec)):
            return None
        if spec.exercise_type is not ExerciseType.AMERICAN:
            return None
        strike = float(spec.strike)
        s_node = float(S[j])
        if spec.option_type is OptionType.CALL:
            intrinsic = s_node - strike
            sign = 1.0
        else:
            intrinsic = strike - s_node
            sign = -1.0
        if intrinsic <= 0.0:
            return None
        # PV at the node should equal intrinsic when in the exercise region.
        # PSOR convergence is exact at the exercise constraint up to the
        # tolerance ``tol``; allow a small relative slack.
        if abs(V[j] - intrinsic) > max(1e-8, 1e-6 * intrinsic):
            return None
        return (sign, 0.0)

    def delta(self) -> float:
        r"""Grid delta via parabolic-Lagrange first derivative at exactly
        ``spot`` (not at the nearest node).

        Collapses to the standard central difference on a uniform grid
        when ``spot`` coincides with a node.

        Short-circuits to ``±1`` when the option is American and the
        spot node sits in the early-exercise region (PV equals intrinsic);
        the PDE stencil is unreliable there because of PSOR oscillations.
        """
        S, V, _, _, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return short_circuit[0]
        return self._grid_delta_at_spot(S, V, j, spot)

    def gamma(self) -> float:
        r"""Grid gamma via cubic-Lagrange second derivative at exactly
        ``spot``, with a parabolic fallback when the cubic stencil
        appears polluted by PDE noise (see ``_grid_gamma_safe``).

        Short-circuits to ``0`` when the option is American and the
        spot node sits in the early-exercise region (PV equals intrinsic).
        """
        S, V, _, _, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return short_circuit[1]
        return self._grid_gamma_safe(S, V, j, spot)

    def _grid_theta_bs_identity(
        self,
        S: np.ndarray,
        V: np.ndarray,
        j: int,
        spot: float,
        last_dtau: float,
    ) -> float:
        r"""Return per-calendar-day theta via the Black-Scholes PDE identity.

        .. math::

            \Theta = r V - (r - q) S \Delta - \tfrac{1}{2} \sigma^{2} S^{2} \Gamma

        ``V`` is interpolated to exact ``spot`` with the same parabolic
        Lagrange stencil used by :meth:`_grid_delta_at_spot`; ``r`` and
        ``q`` are the forward rates over the first PDE step (exact for
        flat curves, first-order local otherwise).  Result is divided by
        365 to match the calendar-day convention used elsewhere.
        """
        # PV at exactly spot (parabolic Lagrange through the delta stencil).
        x0, x1, x2 = S[j - 1], S[j], S[j + 1]
        v0, v1, v2 = V[j - 1], V[j], V[j + 1]
        pv_at_spot = float(
            v0 * (spot - x1) * (spot - x2) / ((x0 - x1) * (x0 - x2))
            + v1 * (spot - x0) * (spot - x2) / ((x1 - x0) * (x1 - x2))
            + v2 * (spot - x0) * (spot - x1) / ((x2 - x0) * (x2 - x1))
        )
        delta = self._grid_delta_at_spot(S, V, j, spot)
        gamma = self._grid_gamma_safe(S, V, j, spot)

        sigma = float(self.underlying.volatility)
        dt_probe = max(last_dtau, 1.0e-8)
        r = float(self.valuation_ctx.discount_curve.forward_rate(0.0, dt_probe))
        q = 0.0
        if self.underlying.dividend_curve is not None:
            q = float(self.underlying.dividend_curve.forward_rate(0.0, dt_probe))

        theta_annual = (
            r * pv_at_spot - (r - q) * spot * delta - 0.5 * sigma * sigma * spot * spot * gamma
        )
        return float(theta_annual / 365.0)

    def theta(self) -> float:
        r"""Grid theta via the Black-Scholes PDE identity.

        Uses :meth:`_grid_theta_bs_identity` so theta attains the same
        order of accuracy as the grid delta and gamma — notably better
        near steep spatial gradients such as a barrier, where a
        first-order backward time-difference amplifies the local error
        in ``V``.  Returned per **calendar day**.

        Short-circuits to ``0`` when the option is American and the spot
        node sits in the early-exercise region; the Black-Scholes PDE
        becomes an inequality there and the identity no longer holds.
        """
        S, V, _, last_dtau, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return 0.0
        if last_dtau <= 0.0:
            return 0.0
        return self._grid_theta_bs_identity(S, V, j, spot, last_dtau)


class _FDValuationBase(_FDGridGreeksMixin):
    """Base class for European/American FD valuation."""

    _early_exercise: bool = False

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying  # type: ignore[assignment]
        assert isinstance(valuation_ctx.params, PDEParams)
        self.pde_params = valuation_ctx.params
        # Lazy PDE-solve cache.  Populated on first ``_solve`` call and
        # shared by every subsequent PV / grid-greek access on this
        # instance, so ``delta`` / ``gamma`` / ``theta`` after a
        # ``present_value`` call are O(1) grid lookups.  Double-checked
        # locking keeps the fast path lock-free on cache hits while
        # guaranteeing at most one expensive solve under concurrent access.
        self._solve_result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        # ``RLock`` so nested caching calls within the same thread don't
        # deadlock (e.g. an engine that dispatches to another cached
        # helper while holding the solve lock).
        self._solve_lock = threading.RLock()

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
        pv, S, V, *_ = self._solve()
        return pv, S, V

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Memoised PDE solve result (see ``_compute_solve`` for the real work)."""
        if self._solve_result is not None:
            return self._solve_result
        with self._solve_lock:
            if self._solve_result is not None:
                return self._solve_result
            self._solve_result = self._compute_solve()
            return self._solve_result

    def _compute_solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the PDE finite-difference solve."""
        params = self.pde_params

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

        time_to_maturity = self.valuation_ctx._maturity_year_fraction()

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
        if isinstance(spec, PayoffSpec):
            custom_payoff = spec.payoff
            custom_boundary_model = spec.boundary_model
        else:
            custom_payoff = None
            custom_boundary_model = None

        return _fd_core(
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
            american_solver=params.american_solver if self._early_exercise else None,
            omega=float(params.omega) if self._early_exercise else None,
            tol=float(params.tol) if self._early_exercise else None,
            max_iter=int(params.max_iter) if self._early_exercise else None,
            payoff_fn=custom_payoff,
            payoff_boundary_model=custom_boundary_model,
        )

    def present_value(self) -> float:
        """Return present value from the PDE solve."""
        label = "PDE American" if self._early_exercise else "PDE European"
        with log_timing(logger, f"{label} present_value", self.pde_params.log_timings):
            pv, *_ = self._solve()
        return float(pv)


class _FDEuropeanValuation(_FDValuationBase):
    """European option valuation using PDE finite differences."""

    _early_exercise = False


class _FDAmericanValuation(_FDValuationBase):
    """American option valuation using PDE finite differences."""

    _early_exercise = True


# ═══════════════════════════════════════════════════════════════════════════
# Barrier option PDE
# ═══════════════════════════════════════════════════════════════════════════


def _barrier_monitoring_taus(
    *,
    monitoring_dates: Sequence[dt.datetime],
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    day_count_convention: DayCountConvention,
) -> list[float]:
    """Convert monitoring datetime schedule to tau-space values.

    Returns sorted list of taus (time remaining from maturity perspective).
    """
    ttm = calculate_year_fraction(
        pricing_date,
        maturity,
        day_count_convention=day_count_convention,
    )
    taus: list[float] = []
    for d in monitoring_dates:
        if pricing_date <= d <= maturity:
            t = calculate_year_fraction(
                pricing_date,
                d,
                day_count_convention=day_count_convention,
            )
            taus.append(round(ttm - t, 12))
    taus.sort()
    return taus


def _build_ko_continuous_log_grid(
    *,
    smin_target: float,
    smax_target: float,
    ref_price: float,
    smax_mult: float,
    direction: BarrierDirection,
    volatility: float,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a log-spot grid with the barrier as a boundary node.

    For explicit-family schemes, preserves Hull's dz scale;
    CN/IMPLICIT honor spot_steps directly.
    """
    explicit_scheme = method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
    if direction is BarrierDirection.DOWN:
        zmin = np.log(smin_target)
        zmax_default = np.log(smax_target)
        if explicit_scheme:
            dz_hull = volatility * np.sqrt(3.0 * (time_to_maturity / time_steps))
            grid_width = spot_steps * dz_hull
            if (zmax_default - zmin) > grid_width:
                dz = (zmax_default - zmin) / spot_steps
            else:
                dz = dz_hull
                zmax_default = zmin + grid_width
        else:
            dz = (zmax_default - zmin) / spot_steps
        Z = np.linspace(zmin, zmax_default, spot_steps + 1)
    else:
        zmax = np.log(smax_target)
        zmin_default = np.log(max(ref_price / smax_mult, 1.0e-8))
        if explicit_scheme:
            dz_hull = volatility * np.sqrt(3.0 * (time_to_maturity / time_steps))
            grid_width = spot_steps * dz_hull
            if (zmax - zmin_default) > grid_width:
                dz = (zmax - zmin_default) / spot_steps
            else:
                dz = dz_hull
                zmin_default = zmax - grid_width
        else:
            dz = (zmax - zmin_default) / spot_steps
        Z = np.linspace(zmin_default, zmax, spot_steps + 1)
    S = np.exp(Z)
    return Z, S, dz


def _fd_barrier_ko_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    barrier: float,
    direction: BarrierDirection,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,
    monitoring_taus: list[float] | None,  # required (not None) for DISCRETE
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise | None = None,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Core finite-difference solver for knock-out barrier options.

    For continuous knock-out barriers, the grid is truncated at the barrier
    level so the barrier becomes a domain boundary.

    For discrete knock-out barriers, the full grid is used and barrier
    resets are applied at monitoring dates (analogous to dividend jumps).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_final, V_prev, last_dtau)``
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS

    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    # ── Grid construction ─────────────────────────────────────────────
    # For continuous KO, truncate the grid at the barrier.
    # For discrete KO, use the standard full grid.
    ref_price = max(spot, strike)

    if continuous:
        if direction is BarrierDirection.DOWN:
            smin_target = barrier
            smax_target = smax_mult * ref_price
        else:
            smin_target = max(ref_price / smax_mult, 1.0e-8)
            smax_target = barrier
    else:
        smin_target = None
        smax_target = None

    if space_grid is PDESpaceGrid.LOG_SPOT:
        if continuous:
            grid, S, dz = _build_ko_continuous_log_grid(
                smin_target=smin_target,
                smax_target=smax_target,
                ref_price=ref_price,
                smax_mult=smax_mult,
                direction=direction,
                volatility=volatility,
                time_to_maturity=time_to_maturity,
                spot_steps=spot_steps,
                time_steps=time_steps,
                method=method,
            )
        else:
            grid, S, dz = _build_log_grid(
                spot=spot,
                strike=strike,
                time_to_maturity=time_to_maturity,
                volatility=volatility,
                smax_mult=smax_mult,
                spot_steps=spot_steps,
                time_steps=time_steps,
                method=method,
                anchor_spot=barrier,
                anchor_half_step=True,
            )
    else:
        # Spot grid
        if continuous:
            grid = np.linspace(smin_target, smax_target, spot_steps + 1)
            S = grid
            dS = (smax_target - smin_target) / spot_steps
        else:
            smax = float(smax_mult * ref_price)
            grid, S, dS = _build_spot_grid(
                smin=0.0,
                smax=smax,
                spot_steps=spot_steps,
                anchor_spot=barrier,
                anchor_half_step=True,
            )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices

    # ── Terminal payoff ───────────────────────────────────────────────
    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    # American exercise intrinsic is the holder's exercise value at any
    # in-life moment, which is the vanilla payoff — independent of any
    # KO-zone modification applied to the maturity payoff below.  Under
    # discrete monitoring the holder can exercise between observations
    # even when sitting in the KO zone, so the PSOR floor must use the
    # unmodified vanilla intrinsic.  (For continuous KO the grid is
    # truncated at the barrier and the KO zone is absent from the grid,
    # so the same vanilla array is also correct on the alive side.)
    intrinsic = payoff.copy() if early_exercise else None

    # For continuous KO: payoff is zero on the barrier side (enforced
    # by grid truncation since the barrier is at the boundary).
    # For discrete KO: zero out the payoff on the knocked-out side at maturity
    # only if maturity is a monitoring date (which it typically is).
    if not continuous:
        assert monitoring_taus is not None  # validated above
        # tau=0 is maturity; if it's a monitoring tau, apply the reset
        if any(abs(tau) < 1e-12 for tau in monitoring_taus):
            if direction is BarrierDirection.DOWN:
                payoff[S <= barrier] = 0.0
            else:
                payoff[S >= barrier] = 0.0

    V = payoff.copy()

    # ── Dividend schedule ─────────────────────────────────────────────
    schedule = dividend_schedule or []
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V[:] = np.maximum(V, payoff)

    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # ── Merge monitoring taus into grid ───────────────────────────────
    dividend_taus = list(dividend_map.keys())
    extra_taus = dividend_taus.copy()
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None  # validated above
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}

    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and space_grid is PDESpaceGrid.SPOT:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=dS if space_grid is PDESpaceGrid.SPOT else (smax - smin) / spot_steps,
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            hull_discounting=method is PDEMethod.EXPLICIT_HULL,
        )

    # ── Time-stepping ─────────────────────────────────────────────────
    df_0T = float(discount_curve.df(time_to_maturity))
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))
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
                dS=(smax - smin) / spot_steps,
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

        # ── Barrier boundary conditions ───────────────────────────────
        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        if continuous:
            # Barrier is at a grid boundary; set its value
            if rebate == 0.0:
                barrier_bv = 0.0
            elif rebate_timing is RebateTiming.AT_HIT:
                barrier_bv = rebate
            else:
                # AT_EXPIRY: discounted from current time to maturity
                barrier_bv = rebate * df_tT

            if direction is BarrierDirection.DOWN:
                # Barrier at left boundary, vanilla far-field at right
                left = barrier_bv
                if option_type is OptionType.PUT:
                    right = 0.0
                else:
                    continuation = smax * dq_tT - strike * df_tT
                    intrinsic_bv = max(smax - strike, 0.0)
                    right = (
                        max(continuation, intrinsic_bv)
                        if early_exercise
                        else max(continuation, 0.0)
                    )
            else:
                # Barrier at right boundary, vanilla far-field at left
                right = barrier_bv
                if option_type is OptionType.PUT:
                    intrinsic_bv = max(strike - smin, 0.0)
                    continuation = strike * df_tT - smin * dq_tT
                    left = max(continuation, intrinsic_bv) if early_exercise else continuation
                else:
                    left = 0.0
        else:
            # Discrete monitoring: standard vanilla boundaries
            left, right = _boundary_values(
                option_type=option_type,
                strike=strike,
                smin=smin,
                smax=smax,
                df_tT=df_tT,
                dq_tT=dq_tT,
                early_exercise=early_exercise,
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

        # ── Discrete dividend jump ────────────────────────────────────
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

        # ── Discrete barrier reset ────────────────────────────────────
        if monitoring_tau_set is not None:
            tau_key = round(tau_curr, 12)
            if tau_key in monitoring_tau_set:
                if rebate == 0.0:
                    reset_val = 0.0
                elif rebate_timing is RebateTiming.AT_HIT:
                    reset_val = rebate
                else:
                    reset_val = rebate * df_tT
                if direction is BarrierDirection.DOWN:
                    V[S <= barrier] = reset_val
                else:
                    V[S >= barrier] = reset_val
                # Re-enforce early exercise on surviving nodes
                if early_exercise and intrinsic is not None:
                    alive = S > barrier if direction is BarrierDirection.DOWN else S < barrier
                    mask = alive & (intrinsic > V)
                    V[mask] = intrinsic[mask]

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE barrier PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V))
    return price, S, V, V_prev, last_dtau


def _subgrid_pde_step(
    V: np.ndarray,
    V_prev: np.ndarray,
    j_sub: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    method: PDEMethod,
    r_dt: float,
) -> np.ndarray:
    """One PDE time-step on a sub-grid. Used for the inactive-surface solve in a coupled KI PDE.

    Parameters
    ----------
    V, V_prev : full-length value arrays (current / previous tau layer).
    j_sub : node indices of the sub-grid interior.
    a, b, c : scaled operator coefficients for the **full** interior
        (``j=1..M-1``); this helper slices them to ``j_sub``.
    left, right : Dirichlet boundary values for the sub-grid.
    method : PDE stepping scheme.
    r_dt : ``r * d_tau`` for Hull explicit discounting (0 otherwise).
    """
    V = V.copy()
    ci: np.ndarray = j_sub - 1  # coefficient indices into full-interior arrays
    a_s, b_s, c_s = a[ci], b[ci], c[ci]

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        interior = -a_s * V_prev[j_sub - 1] + (1.0 - b_s) * V_prev[j_sub] - c_s * V_prev[j_sub + 1]
        V[j_sub] = interior / (1.0 + r_dt)
        V[j_sub[0] - 1] = left
        V[j_sub[-1] + 1] = right
    elif method is PDEMethod.IMPLICIT:
        diag = 1.0 + b_s
        rhs = V_prev[j_sub].copy()
        rhs[0] -= a_s[0] * left
        rhs[-1] -= c_s[-1] * right
        V[j_sub] = _solve_tridiagonal_thomas(a_s[1:], diag, c_s[:-1], rhs)
    else:
        a_h, b_h, c_h = a_s * 0.5, b_s * 0.5, c_s * 0.5
        diag = 1.0 + b_h
        rhs = -a_h * V_prev[j_sub - 1] + (1.0 - b_h) * V_prev[j_sub] - c_h * V_prev[j_sub + 1]
        rhs[0] -= a_h[0] * left
        rhs[-1] -= c_h[-1] * right
        V[j_sub] = _solve_tridiagonal_thomas(a_h[1:], diag, c_h[:-1], rhs)
    return V


def _fd_barrier_ki_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    barrier: float,
    direction: BarrierDirection,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,  # unused for KI but kept for signature consistency with KO core
    monitoring_taus: list[float] | None,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise | None = None,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Two-surface coupled PDE solver for knock-in barrier options.

    Maintains two value surfaces that are stepped backward in time:

    * **Active** (``V_act``): the barrier has been hit; behaves as a standard
      option (with early-exercise projection when ``early_exercise=True``).
    * **Inactive** (``V_inact``): the barrier has not yet been hit; pure
      continuation PDE with no exercise allowed.

    At the barrier the inactive surface is coupled to the active surface
    (state transition, not absorption).

    The option starts in the inactive state, so the price is read from
    ``V_inact`` at the spot level.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_inact_final, V_inact_prev, last_dtau)``
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS

    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    # ── Grid construction (full grid, not truncated) ──────────────────
    ref_price = max(spot, strike)

    if space_grid is PDESpaceGrid.LOG_SPOT:
        grid, S, dz = _build_log_grid(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
            anchor_spot=barrier,
            anchor_half_step=not continuous,
        )
    else:
        smax = float(smax_mult * ref_price)
        grid, S, dS = _build_spot_grid(
            smin=0.0,
            smax=smax,
            spot_steps=spot_steps,
            anchor_spot=barrier,
            anchor_half_step=not continuous,
        )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices

    # ── Terminal payoff ───────────────────────────────────────────────
    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    intrinsic = payoff if early_exercise else None

    df_0T = float(discount_curve.df(time_to_maturity))

    # Active surface: vanilla payoff at maturity
    V_act = payoff.copy()
    # Inactive surface: rebate PV at maturity (0 if no rebate)
    # KI rebate is always AT_EXPIRY (validated by BarrierSpec), so terminal
    # value for inactive paths = undiscounted rebate (we are at maturity).
    V_inact = np.full_like(payoff, rebate)

    # ── Dividend schedule ─────────────────────────────────────────────
    schedule = dividend_schedule or []
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}

    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V_act, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V_act[:] = np.maximum(V_act, payoff)
        # The inactive terminal surface is spot-independent (equal to the
        # maturity rebate everywhere), so the maturity-date dividend jump is
        # a no-op for V_inact.

    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # ── Merge monitoring taus into time grid ──────────────────────────
    dividend_taus = list(dividend_map.keys())
    extra_taus = dividend_taus.copy()
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}

    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

    # ── Barrier index for sub-grid coupling ──────────────────────────
    # Find the grid node closest to the barrier level.
    j_H = int(np.argmin(np.abs(S - barrier)))

    # Terminal coupling: if the barrier is already triggered at maturity,
    # the inactive state must transition immediately to the active payoff.
    if continuous or (monitoring_tau_set is not None and 0.0 in monitoring_tau_set):
        if direction is BarrierDirection.DOWN:
            terminal_hit_mask = S <= barrier
        else:
            terminal_hit_mask = S >= barrier
        V_inact[terminal_hit_mask] = V_act[terminal_hit_mask]

    # For continuous monitoring, the inactive surface PDE is solved only
    # on the far side of the barrier (above H for down-in, below H for
    # up-in), with V_act[j_H] as the inner Dirichlet boundary.  This
    # makes the coupling implicit in the solve and avoids the operator-
    # splitting error that arises from full-grid solve + post-hoc
    # coupling.
    if continuous:
        if direction is BarrierDirection.DOWN:
            # Solve above barrier: interior nodes j_H+1 .. spot_steps-1
            j_inact = np.arange(j_H + 1, spot_steps)
        else:
            # Solve below barrier: interior nodes 1 .. j_H-1
            j_inact = np.arange(1, j_H)
    else:
        j_inact = j  # full interior for discrete (coupling at monitoring dates)

    # ── Time-stepping ────────────────────────────────────────────────
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)

    V_act_prev = V_act.copy()
    V_inact_prev = V_inact.copy()
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

        # Discount factors for boundary conditions
        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        # Vanilla boundary conditions — used for active surface
        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smin=smin,
            smax=smax,
            df_tT=df_tT,
            dq_tT=dq_tT,
            early_exercise=early_exercise,
        )

        V_act_prev = V_act.copy()
        V_inact_prev = V_inact.copy()
        last_dtau = d_tau

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        # ── Step A: PDE step for active surface (with American exercise) ──
        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V_act = _explicit_step(
                V_act_prev,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic,
                r_dt=r * d_tau if hull_discounting else 0.0,
            )
        else:
            V_act, psor_iters = _implicit_cn_step(
                V_act_prev,
                V_act,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                intrinsic,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):  # type: ignore[arg-type]
                    psor_not_converged += 1

        # ── Step B: PDE step for inactive surface (no exercise) ──────────
        # For continuous monitoring the inactive surface is solved on a
        # sub-grid restricted to nodes on the far side of the barrier,
        # with V_act[j_H] as the inner Dirichlet boundary.
        if continuous:
            # Inner BC: V_act at barrier; outer BC: rebate PV (0 if no rebate)
            rebate_bv = rebate * df_tT
            if direction is BarrierDirection.DOWN:
                left_inact = float(V_act[j_H])  # inner (barrier side)
                right_inact = rebate_bv  # far side → 0 if no rebate
            else:
                left_inact = rebate_bv  # far side → 0 if no rebate
                right_inact = float(V_act[j_H])  # inner (barrier side)

            if j_inact.size > 0:
                V_inact = _subgrid_pde_step(
                    V_inact,
                    V_inact_prev,
                    j_inact,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    method_used,
                    r_dt=r * d_tau if hull_discounting else 0.0,
                )

            # The sub-grid solve fills only the continuation-region interior.
            # The assignments below complete the current V_inact slice over
            # the full spatial grid by imposing the hit-side coupling region
            # and the far-field boundary.
            if direction is BarrierDirection.DOWN:
                V_inact[: j_H + 1] = V_act[: j_H + 1]
                V_inact[-1] = rebate_bv  # far-field boundary
            else:
                V_inact[j_H:] = V_act[j_H:]
                V_inact[0] = rebate_bv
        else:
            # Discrete monitoring: full grid solve, coupling at monitoring dates
            # For an inactive KI, on the safe far side its asymptotic value is
            # the no-hit value (rebate PV, or 0 if no rebate). On the risky
            # side, immediate coupling to the current active boundary is too
            # aggressive between monitoring dates because activation cannot
            # occur until the next observation. Use a one-step look-ahead
            # proxy: current active boundary on monitoring dates, otherwise the
            # next-time-slice active boundary carried in V_act_prev. This is a
            # pragmatic discrete-monitoring closure, not an exact asymptotic
            # boundary condition.
            rebate_bv = rebate * df_tT
            tau_key = round(tau_curr, 12)
            is_monitoring_step = monitoring_tau_set is not None and tau_key in monitoring_tau_set
            if direction is BarrierDirection.DOWN:
                left_inact = float(V_act[0] if is_monitoring_step else V_act_prev[0])
                right_inact = rebate_bv
            else:
                left_inact = rebate_bv
                right_inact = float(V_act[-1] if is_monitoring_step else V_act_prev[-1])
            if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
                V_inact = _explicit_step(
                    V_inact_prev,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    None,
                    r_dt=r * d_tau if hull_discounting else 0.0,
                )
            else:
                V_inact, _ = _implicit_cn_step(
                    V_inact_prev,
                    V_inact,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    method_used,
                    None,
                    american_solver,
                    omega,
                    tol,
                    max_iter,
                )

            # Discrete barrier coupling at monitoring dates.
            # This is imposed on the ex-div surface at the observation time;
            # the dividend jump below then carries that knocked-in state back
            # to the pre-div surface (for example, cum-div spots that fall
            # through the barrier once the cash dividend goes ex).
            if monitoring_tau_set is not None:
                if tau_key in monitoring_tau_set:
                    if direction is BarrierDirection.DOWN:
                        mask = S <= barrier
                    else:
                        mask = S >= barrier
                    V_inact[mask] = V_act[mask]

        # ── Discrete dividend jumps (both surfaces) ─────────────────────
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V_act, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V_act[:] = np.maximum(V_act, intrinsic)
                _apply_dividend_jump(V_inact, grid, amount, space_grid=space_grid)
                if continuous:
                    if direction is BarrierDirection.DOWN:
                        V_inact[: j_H + 1] = V_act[: j_H + 1]
                    else:
                        V_inact[j_H:] = V_act[j_H:]

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE barrier KI PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    # Price from inactive surface (option starts inactive)
    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V_inact))
    return price, S, V_inact, V_inact_prev, last_dtau


class _FDBarrierValuation(_FDGridGreeksMixin):
    """PDE finite-difference valuation for barrier options.

    Supports:
    - Continuous and discrete knock-out (European and American)
    - Continuous and discrete knock-in via in-out parity (European only)
    - American knock-in via two-surface coupled PDE
    - Rebates (at-hit and at-expiry)
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying  # type: ignore[assignment]
        self._spec: BarrierSpec = valuation_ctx.spec  # type: ignore[assignment]
        assert isinstance(valuation_ctx.params, PDEParams)
        self.pde_params = valuation_ctx.params
        # Lazy PDE-solve caches.  ``_solve_result`` holds the full 5-tuple
        # returned by the backward PDE solve (KO directly; EU KI via
        # parity; AM KI via the two-surface coupled solver).
        # ``_ki_components_result`` holds the raw KO + vanilla solves used
        # by European-KI parity greeks.  Both are populated on first
        # demand and shared across every subsequent PV / greek call on
        # this instance.
        self._solve_result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        self._ki_components_result: (
            tuple[
                tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
                tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
            ]
            | None
        ) = None
        # Separate ``RLock``s for the two independent caches.  European
        # KI ``_compute_solve`` internally calls
        # ``_solve_european_ki_components`` which acquires a different
        # lock — no deadlock risk there — but we still use re-entrant
        # locks to future-proof against nested cached calls on the same
        # lock within a single thread.
        self._solve_lock = threading.RLock()
        self._ki_components_lock = threading.RLock()

    def _resolved_knock_out_value(self) -> float | None:
        if (
            self._spec.action is not BarrierAction.OUT
            or not self.valuation_ctx._barrier_triggered_at_inception()
        ):
            return None

        if self._spec.rebate <= 0.0:
            return 0.0
        if self._spec.rebate_timing is RebateTiming.AT_HIT:
            return float(self._spec.rebate)

        ttm = self.valuation_ctx._maturity_year_fraction()
        return float(self._spec.rebate) * float(self.valuation_ctx.discount_curve.df(ttm))

    def _last_dtau(self) -> float:
        solve_args = self._base_solve_args()
        time_to_maturity = float(solve_args["time_to_maturity"])
        dividend_taus = [
            tau
            for tau, _ in solve_args["dividend_schedule"] or []
            if 1.0e-12 < tau < time_to_maturity - 1.0e-12
        ]
        extra_taus = dividend_taus.copy()
        if solve_args["monitoring_taus"] is not None:
            extra_taus.extend(solve_args["monitoring_taus"])
        tau_grid = _build_tau_grid(time_to_maturity, int(solve_args["time_steps"]), extra_taus)
        if tau_grid.size < 2:
            return 0.0
        return float(tau_grid[-1] - tau_grid[-2])

    def _discounted_rebate_theta(self, last_dtau: float) -> float:
        if (
            self._spec.rebate <= 0.0
            or self._spec.rebate_timing is not RebateTiming.AT_EXPIRY
            or last_dtau <= 0.0
        ):
            return 0.0

        ttm = self.valuation_ctx._maturity_year_fraction()
        discount_curve = self.valuation_ctx.discount_curve
        current_value = float(self._spec.rebate) * float(discount_curve.df(ttm))
        previous_value = float(self._spec.rebate) * float(
            discount_curve.df(max(ttm - last_dtau, 0.0))
        )
        return float((previous_value - current_value) / last_dtau / 365.0)

    def _resolved_knock_out_theta(self) -> float:
        return self._discounted_rebate_theta(self._last_dtau())

    @staticmethod
    def _grid_delta_from_result(
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, _ = result
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return _FDGridGreeksMixin._grid_delta_at_spot(S, V, j, spot)

    @staticmethod
    def _grid_gamma_from_result(
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, _ = result
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return _FDGridGreeksMixin._grid_gamma_safe(S, V, j, spot)

    def _grid_theta_from_result(
        self,
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, last_dtau = result
        if last_dtau <= 0.0:
            return 0.0
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return self._grid_theta_bs_identity(S, V, j, spot, last_dtau)

    def _base_solve_args(self) -> dict:
        """Build keyword arguments shared by both KO and KI solvers."""
        params = self.pde_params

        spec = self._spec
        ctx = self.valuation_ctx
        early_exercise = spec.exercise_type is ExerciseType.AMERICAN

        time_to_maturity = ctx._maturity_year_fraction()

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=self.underlying.discrete_dividends,
            pricing_date=ctx.pricing_date,
            maturity=ctx.maturity,
            day_count_convention=ctx.day_count_convention,
        )
        # Resolve monitoring dates to taus
        monitoring_taus: list[float] | None = None
        if spec.monitoring is BarrierMonitoring.DISCRETE:
            mon_dates = ctx._barrier_monitoring_dates()
            monitoring_taus = _barrier_monitoring_taus(
                monitoring_dates=mon_dates,
                pricing_date=ctx.pricing_date,
                maturity=ctx.maturity,
                day_count_convention=ctx.day_count_convention,
            )

        args = dict(
            spot=float(self.underlying.initial_value),
            strike=float(spec.strike),
            time_to_maturity=float(time_to_maturity),
            volatility=float(self.underlying.volatility),
            discount_curve=ctx.discount_curve,
            dividend_curve=self.underlying.dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=spec.option_type,
            barrier=float(spec.barrier),
            direction=spec.direction,
            monitoring=spec.monitoring,
            rebate=float(spec.rebate),
            rebate_timing=spec.rebate_timing,
            monitoring_taus=monitoring_taus,
            smax_mult=float(params.smax_mult),
            spot_steps=int(params.spot_steps),
            time_steps=int(params.time_steps),
            early_exercise=early_exercise,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
            american_solver=params.american_solver if early_exercise else None,
            omega=float(params.omega) if early_exercise else None,
            tol=float(params.tol) if early_exercise else None,
            max_iter=int(params.max_iter) if early_exercise else None,
        )
        return args

    def _solve_european_ki_components(
        self,
    ) -> tuple[
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
    ]:
        """Return the native KO and vanilla solves used by European KI parity.

        Memoised on the instance: the first call runs both solves; every
        subsequent call (e.g. from :meth:`delta`, :meth:`gamma`,
        :meth:`theta`, or internally from :meth:`_solve`) is O(1).
        """
        if self._ki_components_result is not None:
            return self._ki_components_result
        with self._ki_components_lock:
            if self._ki_components_result is not None:
                return self._ki_components_result
            self._ki_components_result = self._compute_european_ki_components()
            return self._ki_components_result

    def _compute_european_ki_components(
        self,
    ) -> tuple[
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
    ]:
        solve_args = self._base_solve_args()
        ko_result = _fd_barrier_ko_core(**solve_args)
        van_result = _fd_core(
            spot=solve_args["spot"],
            strike=solve_args["strike"],
            time_to_maturity=solve_args["time_to_maturity"],
            volatility=solve_args["volatility"],
            discount_curve=solve_args["discount_curve"],
            dividend_curve=solve_args["dividend_curve"],
            dividend_schedule=solve_args["dividend_schedule"],
            option_type=self._spec.option_type,
            smax_mult=solve_args["smax_mult"],
            spot_steps=solve_args["spot_steps"],
            time_steps=solve_args["time_steps"],
            early_exercise=False,
            method=solve_args["method"],
            rannacher_steps=solve_args["rannacher_steps"],
            space_grid=solve_args["space_grid"],
        )
        return ko_result, van_result

    def _is_european_ki(self) -> bool:
        """True iff this spec is a European knock-in — the only case that
        needs native-surface parity (V_KI = V_vanilla − V_KO + rebate leg)
        rather than the mixin's direct grid extraction."""
        spec = self._spec
        return spec.action is BarrierAction.IN and spec.exercise_type is ExerciseType.EUROPEAN

    def delta(self) -> float:
        spec = self._spec
        if self.valuation_ctx._barrier_triggered_at_inception():
            if spec.action is BarrierAction.OUT:
                return 0.0
            return self.valuation_ctx._vanilla_equivalent_valuation().delta()

        if self._is_european_ki():
            ko_result, van_result = self._solve_european_ki_components()
            spot = float(self.underlying.initial_value)
            return self._grid_delta_from_result(van_result, spot) - self._grid_delta_from_result(
                ko_result, spot
            )

        return super().delta()

    def gamma(self) -> float:
        """Return grid gamma, using native-surface parity for European KI barriers."""
        spec = self._spec
        if self.valuation_ctx._barrier_triggered_at_inception():
            if spec.action is BarrierAction.OUT:
                return 0.0
            return self.valuation_ctx._vanilla_equivalent_valuation().gamma()

        if self._is_european_ki():
            ko_result, van_result = self._solve_european_ki_components()
            spot = float(self.underlying.initial_value)
            return self._grid_gamma_from_result(van_result, spot) - self._grid_gamma_from_result(
                ko_result, spot
            )

        return super().gamma()

    def theta(self) -> float:
        spec = self._spec
        if self.valuation_ctx._barrier_triggered_at_inception():
            if spec.action is BarrierAction.OUT:
                return self._resolved_knock_out_theta()
            return self.valuation_ctx._vanilla_equivalent_valuation().theta()

        if self._is_european_ki():
            ko_result, van_result = self._solve_european_ki_components()
            spot = float(self.underlying.initial_value)
            ko_theta = self._grid_theta_from_result(ko_result, spot)
            vanilla_theta = self._grid_theta_from_result(van_result, spot)
            rebate_theta = self._discounted_rebate_theta(ko_result[-1])
            return vanilla_theta + rebate_theta - ko_theta

        return super().theta()

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Memoised PDE solve result.

        The first call runs the backward solve (KO directly, American KI
        via the coupled two-surface solver, European KI via parity on the
        cached KO + vanilla components).  Every subsequent call on the
        same instance — including those triggered transparently by the
        grid-greek mixin's :meth:`delta`, :meth:`gamma`, :meth:`theta` —
        is an O(1) tuple lookup.
        """
        if self._solve_result is not None:
            return self._solve_result
        with self._solve_lock:
            if self._solve_result is not None:
                return self._solve_result
            self._solve_result = self._compute_solve()
            return self._solve_result

    def _compute_solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the PDE solve, handling KI via parity or coupled PDE."""
        spec = self._spec
        solve_args = self._base_solve_args()

        if spec.action is BarrierAction.OUT:
            return _fd_barrier_ko_core(**solve_args)

        # American knock-in: two-surface coupled PDE
        if spec.exercise_type is ExerciseType.AMERICAN:
            return _fd_barrier_ki_core(**solve_args)

        # European knock-in via parity: V_KI = V_vanilla + R * df_T - V_KO
        # When R=0 this reduces to V_vanilla - V_KO.
        ko_result, van_result = self._solve_european_ki_components()
        ko_price, S_ko, V_ko, V_ko_prev, last_dtau_ko = ko_result
        van_price, S_van, V_van, V_van_prev, _ = van_result

        ttm = float(solve_args["time_to_maturity"])
        discount_curve = solve_args["discount_curve"]
        df_T = float(discount_curve.df(ttm))
        ki_price = van_price + spec.rebate * df_T - ko_price

        if last_dtau_ko > 0.0:
            previous_ttm = max(ttm - last_dtau_ko, 0.0)
            rebate_prev = float(spec.rebate) * float(discount_curve.df(previous_ttm))
        else:
            rebate_prev = float(spec.rebate) * df_T

        # For grid greeks we return the KO grid (best we can do). Include the
        # KI no-hit rebate PV in the reconstructed grids as well as the scalar price.
        V_ki = np.interp(S_ko, S_van, V_van) + float(spec.rebate) * df_T - V_ko
        V_ki_prev = np.interp(S_ko, S_van, V_van_prev) + rebate_prev - V_ko_prev
        return ki_price, S_ko, V_ki, V_ki_prev, last_dtau_ko

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution."""
        pv, S, V, *_ = self._solve()
        return pv, S, V

    def present_value(self) -> float:
        """Return present value from the PDE barrier solve."""
        if self.valuation_ctx._barrier_triggered_at_inception():
            if self._spec.action is BarrierAction.OUT:
                triggered_value = self._resolved_knock_out_value()
                if triggered_value is None:
                    raise ConfigurationError("Resolved knock-out state unexpectedly unavailable")
                return triggered_value
            return self.valuation_ctx._vanilla_equivalent_valuation().present_value()
        spec = self._spec
        label = f"PDE barrier {'American' if spec.exercise_type is ExerciseType.AMERICAN else 'European'}"
        with log_timing(logger, f"{label} present_value", self.pde_params.log_timings):
            pv, *_ = self._solve()
        return float(pv)
