"""Interest-rate and discount-curve utilities."""

from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np

from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Deterministic discount curve with log-linear interpolation.

    times are year fractions and must be strictly increasing.
    dfs are positive discount factors, typically with df(0)=1.
    Values > 1 are permitted (negative rates) but trigger a warning.
    """

    times: np.ndarray
    dfs: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        df = np.asarray(self.dfs, dtype=float)
        if t.ndim != 1 or df.ndim != 1 or t.shape != df.shape:
            raise ValidationError("times and dfs must be 1D arrays of the same length")
        if t.size < 2:
            raise ValidationError("DiscountCurve requires at least 2 data points")
        if np.any(np.diff(t) <= 0.0):
            raise ValidationError("times must be strictly increasing")
        if np.any(df <= 0.0):
            raise ValidationError("discount factors must be positive")
        if np.any(df > 1.0 + 1e-12):
            warnings.warn(
                "Discount factors > 1 detected (negative rates)",
                stacklevel=2,
            )
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dfs", df)

    @property
    def flat_rate(self) -> float | None:
        """Return the flat continuously-compounded rate if the curve is flat, else ``None``."""
        if self.times.size < 2:
            return None
        # Infer rate from the last tenor: r = -ln(DF) / t
        t_last = float(self.times[-1])
        if t_last <= 0.0:
            return None
        candidate = -float(np.log(self.dfs[-1])) / t_last
        implied = np.exp(-candidate * self.times)
        if np.allclose(self.dfs, implied, rtol=1e-10, atol=1e-12):
            return candidate
        return None

    @classmethod
    def from_forwards(
        cls,
        times: np.ndarray,
        forwards: np.ndarray,
    ) -> DiscountCurve:
        """Build a curve from piecewise-constant forward rates.

        Parameters
        ----------
        times
            Year-fraction grid including 0.  Shape ``(N+1,)``.
        forwards
            Continuously-compounded forward rate on each interval.  Shape ``(N,)``.
        """
        times = np.asarray(times, dtype=float)
        forwards = np.asarray(forwards, dtype=float)
        if times.ndim != 1 or forwards.ndim != 1:
            raise ValidationError("times and forwards must be 1-D arrays")
        if times.size < 2:
            raise ValidationError("times must include at least [0, T]")
        if forwards.size != times.size - 1:
            raise ValidationError("forwards must have length len(times) - 1")
        if not np.isclose(times[0], 0.0):
            raise ValidationError("times must start at 0.0")
        dt_steps = np.diff(times)
        cum_rate = np.concatenate([[0.0], np.cumsum(forwards * dt_steps)])
        dfs = np.exp(-cum_rate)
        return cls(times=times, dfs=dfs)

    @classmethod
    def from_zero_rates(
        cls,
        times: np.ndarray,
        zero_rates: np.ndarray,
    ) -> DiscountCurve:
        """Build a curve from continuously-compounded zero (spot) rates.

        Parameters
        ----------
        times
            Year-fraction grid.  Shape ``(N,)``.
            Must be strictly increasing.  When ``times[0] = 0``, the
            corresponding rate is cosmetic (DF is always 1 there).
        zero_rates
            Continuously-compounded zero rates at each time.  Shape ``(N,)``.
        """
        times = np.asarray(times, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        if times.ndim != 1 or zero_rates.ndim != 1:
            raise ValidationError("times and zero_rates must be 1-D arrays")
        if times.size != zero_rates.size:
            raise ValidationError("times and zero_rates must have the same length")
        if times.size < 1:
            raise ValidationError("times must contain at least one tenor")
        dfs = np.exp(-zero_rates * times)
        return cls(times=times, dfs=dfs)

    @classmethod
    def flat(
        cls,
        rate: float,
        end_time: float,
        steps: int = 1,
    ) -> DiscountCurve:
        """Build a flat continuously-compounded discount curve.

        Parameters
        ----------
        rate
            Flat continuously-compounded annual rate.
        end_time
            Final maturity in years.
        steps
            Number of intervals used to discretize ``[0, end_time]``.

        Returns
        -------
        DiscountCurve
            Discount curve consistent with the supplied flat rate.
        """
        if end_time <= 0.0:
            raise ValidationError("end_time must be positive")
        if steps < 1:
            raise ValidationError("steps must be >= 1")
        times = np.linspace(0.0, float(end_time), int(steps) + 1)
        dfs = np.exp(-float(rate) * times)
        return cls(times=times, dfs=dfs)

    def bump_parallel_zero_rate(self, bump: float) -> DiscountCurve:
        """Return a new curve with a parallel shift to continuously-compounded zero rates.

        Parameters
        ----------
        bump
            Additive shift applied to the zero rate at every tenor.
        """
        bumped_dfs = self.dfs * np.exp(-bump * self.times)
        return DiscountCurve(times=self.times.copy(), dfs=bumped_dfs)

    def df(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate discount factors with log-linear interpolation.

        Parameters
        ----------
        t
            Scalar or array of year fractions.

        Returns
        -------
        np.ndarray
            Interpolated discount factors.
        """
        t = np.asarray(t, dtype=float)
        is_origin = np.isclose(t, 0.0, rtol=0.0, atol=1e-12)
        t_min, t_max = float(self.times[0]), float(self.times[-1])
        outside = ((t < t_min) & ~is_origin) | (t > t_max)
        if np.any(outside):
            warnings.warn(
                f"Extrapolating discount curve outside "
                f"[{t_min:.4f}, {t_max:.4f}] — flat log-DF assumed",
                stacklevel=2,
            )
        log_df = np.log(self.dfs)
        out = np.interp(t, self.times, log_df, left=log_df[0], right=log_df[-1])
        dfs = np.exp(out)
        if np.any(is_origin):
            dfs = np.where(is_origin, 1.0, dfs)
        return dfs

    def forward_rate(self, t0: float, t1: float) -> float:
        """Continuously-compounded forward rate on the interval ``[t0, t1]``.

        Parameters
        ----------
        t0
            Start of the interval (year fraction).
        t1
            End of the interval (year fraction, must be > t0).

        Returns
        -------
        float
            Forward rate ``f`` such that ``D(t1) = D(t0) · exp(−f · (t1 − t0))``.
        """
        if t1 <= t0:
            raise ValidationError("Need t1 > t0")
        df0 = float(self.df(t0))
        df1 = float(self.df(t1))
        return (np.log(df0) - np.log(df1)) / (t1 - t0)

    def step_forward_rates(self, grid: np.ndarray) -> np.ndarray:
        """Forward rates for each consecutive interval of a time grid.

        Parameters
        ----------
        grid
            Strictly increasing array of year fractions (length *n*).

        Returns
        -------
        np.ndarray
            Array of length *n − 1* with the piecewise-constant forward
            rate on each ``[grid[i], grid[i+1]]`` interval.
        """
        grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(grid) <= 0.0):
            raise ValidationError("grid must be strictly increasing")
        df_grid = self.df(grid)
        dt = np.diff(grid)
        return (np.log(df_grid[:-1]) - np.log(df_grid[1:])) / dt
