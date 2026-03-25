"""Binomial-tree valuation engines (Cox-Ross-Rubinstein).

Implements European and American vanilla option pricing, plus Asian-option
extensions used by the core dispatcher.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np
import pandas as pd
from ..enums import AsianAveraging, ExerciseType, OptionType
from ..utils import calculate_year_fraction, pv_discrete_dividends, log_timing
from ..exceptions import (
    ArbitrageViolationError,
    ConfigurationError,
    NumericalError,
    UnsupportedFeatureError,
    ValidationError,
)
from .params import BinomialParams

if TYPE_CHECKING:
    from .core import AsianSpec, OptionValuation, UnderlyingData


logger = logging.getLogger(__name__)


class _BinomialValuationBase:
    """Base class for binomial tree option valuation."""

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying: UnderlyingData = valuation_ctx.underlying  # type: ignore[assignment]
        if not isinstance(valuation_ctx.params, BinomialParams):
            raise ConfigurationError(
                "Binomial valuation requires BinomialParams on OptionValuation"
            )
        self.binom_params: BinomialParams = valuation_ctx.params

    def _setup_binomial_parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Setup binomial tree parameters and lattice.


        Returns
        -------
        tuple of (discount_factors, p, spot_lattice)
            discount_factors : shape (num_steps,) — per-step discount factors
            p : shape (num_steps,) — risk-neutral up-move probabilities
            spot_lattice : shape (num_steps+1, num_steps+1) — CRR spot prices
        """
        num_steps = int(self.binom_params.num_steps)
        start = self.valuation_ctx.pricing_date
        end = self.valuation_ctx.maturity
        time_intervals = pd.date_range(start, end, periods=num_steps + 1)

        # Numerical Parameters
        time_grid = np.array(
            [
                calculate_year_fraction(
                    start,
                    t,
                    day_count_convention=self.valuation_ctx.day_count_convention,
                )
                for t in time_intervals
            ],
            dtype=float,
        )
        dt_steps = np.diff(time_grid)
        if not np.allclose(dt_steps, dt_steps[0]):
            raise UnsupportedFeatureError(
                f"Binomial trees (CRR) require equal time steps, but the "
                f"'{self.valuation_ctx.day_count_convention.value}' day-count convention "
                f"produces non-uniform year fractions on a calendar date grid. "
                f"Use an actual-day convention (ACT/365F, ACT/360) or switch to "
                f"PDE / Monte Carlo pricing."
            )
        delta_t = float(dt_steps[0])

        sigma = float(self.underlying.volatility)
        u = np.exp(sigma * np.sqrt(delta_t))
        d = 1.0 / u

        discount_curve = self.valuation_ctx.discount_curve
        forward_rates = discount_curve.step_forward_rates(time_grid)

        dividend_curve = self.underlying.dividend_curve
        if dividend_curve is not None:
            dividend_forwards = dividend_curve.step_forward_rates(time_grid)
        else:
            dividend_forwards = np.zeros(num_steps, dtype=float)

        growth = np.exp((forward_rates - dividend_forwards) * delta_t)
        too_low = growth <= d
        too_high = growth >= u
        if np.any(too_low) or np.any(too_high):
            parts = []
            if np.any(too_low):
                parts.append(f"exp((r-q)*dt) <= d at {int(np.sum(too_low))} step(s)")
            if np.any(too_high):
                parts.append(f"exp((r-q)*dt) >= u at {int(np.sum(too_high))} step(s)")
            raise ArbitrageViolationError(
                "No-arbitrage condition d < exp((r-q)*dt) < u violated: " + "; ".join(parts)
            )

        p = (growth - d) / (u - d)

        spot_lattice = self._build_spot_lattice(
            num_steps=num_steps,
            time_intervals=time_intervals,
            up=u,
        )

        discount_factors = np.exp(-forward_rates * delta_t)
        return discount_factors, p, spot_lattice

    def _build_spot_lattice(
        self,
        *,
        num_steps: int,
        time_intervals: pd.DatetimeIndex,
        up: float,
    ) -> np.ndarray:
        """Build a CRR spot lattice with time on columns (row=down moves, col=time).

        The down factor is ``1/up`` — the CRR recombining-tree.
        """
        down = 1.0 / up
        spot = float(self.underlying.initial_value)
        discrete_dividends = self.underlying.discrete_dividends
        discount_curve = self.valuation_ctx.discount_curve

        i_idx = np.arange(num_steps + 1)[:, None]
        t_idx = np.arange(num_steps + 1)[None, :]
        up_pow = t_idx - i_idx
        down_pow = i_idx

        if discrete_dividends:
            pv_all = pv_discrete_dividends(
                discrete_dividends,
                curve_date=time_intervals[0],
                end_date=time_intervals[-1],
                discount_curve=discount_curve,
                day_count_convention=self.valuation_ctx.day_count_convention,
                include_start=True,
            )
            spot = max(spot - pv_all, 0.0)

        lattice = spot * (up**up_pow) * (down**down_pow)

        if not discrete_dividends:
            return lattice

        # Escrowed-dividend add-back: at each time step, restore the PV of
        # dividends that have NOT yet gone ex.  Use include_start=False so
        # a dividend going ex at exactly *t* is treated as already paid.
        pv_remaining = np.array(
            [
                pv_discrete_dividends(
                    discrete_dividends,
                    curve_date=time_intervals[0],
                    end_date=time_intervals[-1],
                    discount_curve=discount_curve,
                    start_date=t,
                    day_count_convention=self.valuation_ctx.day_count_convention,
                    include_start=False,
                )
                for t in time_intervals
            ],
            dtype=float,
        )
        lattice += pv_remaining[None, :]
        return lattice

    def _get_intrinsic_values(self, instrument_values: np.ndarray) -> np.ndarray:
        """Calculate intrinsic values at each node.

        Parameters
        ----------
        instrument_values
            Underlying prices at each node.

        Returns
        -------
        np.ndarray
            Intrinsic payoff values.
        """
        K = self.valuation_ctx.strike
        if self.valuation_ctx.option_type in (OptionType.CALL, OptionType.PUT):
            if self.valuation_ctx.option_type is OptionType.CALL:
                return np.maximum(instrument_values - K, 0)
            return np.maximum(K - instrument_values, 0)

        payoff_fn = self.valuation_ctx.spec.payoff  # type: ignore[union-attr]
        return payoff_fn(instrument_values)

    def _solve_backward(self, *, early_exercise: bool) -> np.ndarray:
        """Run CRR backward induction, optionally with early exercise.

        Parameters
        ----------
        early_exercise
            If True, option values are projected up to intrinsic at every
            step (American exercise).

        Returns
        -------
        np.ndarray
            Option value lattice, shape ``(num_steps+1, num_steps+1)``.
        """
        num_steps = int(self.binom_params.num_steps)
        discount_factors, p, spot_lattice = self._setup_binomial_parameters()

        option_lattice = np.zeros_like(spot_lattice)
        intrinsic = self._get_intrinsic_values(spot_lattice)
        option_lattice[:, num_steps] = intrinsic[:, num_steps]

        for t in range(num_steps - 1, -1, -1):
            continuation = (
                p[t] * option_lattice[: t + 1, t + 1]
                + (1 - p[t]) * option_lattice[1 : t + 2, t + 1]
            ) * discount_factors[t]
            if early_exercise:
                option_lattice[: t + 1, t] = np.maximum(intrinsic[: t + 1, t], continuation)
            else:
                option_lattice[: t + 1, t] = continuation

        return option_lattice

    # ------------------------------------------------------------------
    # Tree Greeks (Hull Ch. 21)
    # ------------------------------------------------------------------

    def _tree_greeks_data(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (option_lattice, spot_lattice, delta_t) for tree Greek extraction.

        Raises UnsupportedFeatureError for Asian or other non-vanilla specs
        whose ``solve()`` does not return a 2-D option lattice.
        """
        num_steps = int(self.binom_params.num_steps)
        _, _, spot_lattice = self._setup_binomial_parameters()
        option_lattice = self.solve()  # type: ignore[attr-defined]
        if not isinstance(option_lattice, np.ndarray) or option_lattice.ndim != 2:
            raise UnsupportedFeatureError(
                "Tree Greeks are only available for vanilla binomial options, "
                "not Asian or other path-dependent specs."
            )
        T = calculate_year_fraction(
            self.valuation_ctx.pricing_date,
            self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )
        dt = T / num_steps
        return option_lattice, spot_lattice, dt

    def delta(self) -> float:
        """Extract delta from the binomial tree (Hull Ch. 21).

        .. math::

            \\Delta = \\frac{f_u - f_d}{S_u - S_d}

        where :math:`f_u, f_d` are the option values and :math:`S_u, S_d`
        the spot prices at step 1.
        """
        f, S, _ = self._tree_greeks_data()
        return float((f[0, 1] - f[1, 1]) / (S[0, 1] - S[1, 1]))

    def gamma(self) -> float:
        """Extract gamma from the binomial tree (Hull Ch. 21).

        Uses the three nodes at step 2:

        .. math::

            \\Gamma = \\frac{\\Delta_+ - \\Delta_-}{h}

        where :math:`\\Delta_+ = (f_{uu}-f_{ud})/(S_{uu}-S_{ud})`,
        :math:`\\Delta_- = (f_{ud}-f_{dd})/(S_{ud}-S_{dd})`,
        and :math:`h = (S_{uu}-S_{dd})/2`.
        """
        if self.binom_params.num_steps < 2:
            raise ValidationError("Tree gamma requires num_steps >= 2.")
        f, S, _ = self._tree_greeks_data()
        delta_up = (f[0, 2] - f[1, 2]) / (S[0, 2] - S[1, 2])
        delta_down = (f[1, 2] - f[2, 2]) / (S[1, 2] - S[2, 2])
        h = (S[0, 2] - S[2, 2]) / 2.0
        return float((delta_up - delta_down) / h)

    def theta(self) -> float:
        """Extract theta from the binomial tree (Hull Ch. 21).

        .. math::

            \\Theta = \\frac{f_{ud} - f_0}{2\\Delta t}

        where :math:`f_{ud}` is the central node at step 2 and :math:`f_0`
        the root value.  Returned per **calendar day** (divided by 365).
        """
        if self.binom_params.num_steps < 2:
            raise ValidationError("Tree theta requires num_steps >= 2.")
        f, _, dt = self._tree_greeks_data()
        theta_per_year = (f[1, 2] - f[0, 0]) / (2.0 * dt)
        return float(theta_per_year / 365.0)


class _BinomialEuropeanValuation(_BinomialValuationBase):
    """Implementation of European option valuation using binomial tree."""

    def solve(self) -> np.ndarray:
        """Compute the option value lattice using a binomial tree."""
        logger.debug("Binomial European num_steps=%d", self.binom_params.num_steps)
        return self._solve_backward(early_exercise=False)

    def present_value(self) -> float:
        """Return PV using binomial tree method."""
        with log_timing(logger, "Binomial European present_value", self.binom_params.log_timings):
            option_value_matrix = self.solve()
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialAmericanValuation(_BinomialValuationBase):
    """Implementation of American option valuation using binomial tree."""

    def solve(self) -> np.ndarray:
        """Compute the option value lattice using a binomial tree with early exercise."""
        logger.debug("Binomial American num_steps=%d", self.binom_params.num_steps)
        return self._solve_backward(early_exercise=True)

    def present_value(self) -> float:
        """Return PV using binomial tree method with American early exercise."""
        with log_timing(logger, "Binomial American present_value", self.binom_params.log_timings):
            option_value_matrix = self.solve()
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialAsianValuation(_BinomialValuationBase):
    """Asian option valuation using binomial MC sampling or Hull's representative averages."""

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        super().__init__(valuation_ctx)
        self.spec: AsianSpec = valuation_ctx.spec  # type: ignore[assignment]

    def _tree_dates(self, num_steps: int) -> np.ndarray:
        """Return CRR tree dates (length ``num_steps + 1``)."""
        return np.array(
            pd.date_range(
                self.valuation_ctx.pricing_date,
                self.valuation_ctx.maturity,
                periods=num_steps + 1,
            ).to_pydatetime()
        )

    def _observation_indices(self, num_steps: int) -> np.ndarray:
        """Map contractual fixing dates to nearest CRR tree indices."""
        fixing_dates = self.valuation_ctx._asian_fixing_dates()
        tree_dates = self._tree_dates(num_steps)

        tree_seconds = np.array([d.timestamp() for d in tree_dates], dtype=float)
        mapped: list[int] = []
        for d in fixing_dates:
            idx = int(np.argmin(np.abs(tree_seconds - d.timestamp())))
            mapped.append(idx)

        if not mapped:
            raise ValidationError("No valid Asian observation nodes mapped to the binomial tree.")
        return np.array(mapped, dtype=int)

    def _solve_mc(self) -> np.ndarray:
        """Price Asian option via Monte Carlo sampling on the binomial tree.

        Returns
        -------
        np.ndarray
            Discounted pathwise payoffs.
        """
        if self.valuation_ctx.spec.exercise_type is ExerciseType.AMERICAN:
            raise UnsupportedFeatureError(
                "American early exercise is not supported for binomial Asian MC. "
                "Use Hull representative-average mode instead. Specify asian_tree_averages and "
                "set mc_paths to None in BinomialParams."
            )

        num_steps = int(self.binom_params.num_steps)
        logger.debug(
            "Binomial Asian MC num_steps=%d paths=%s", num_steps, self.binom_params.mc_paths
        )
        discount_factors, p, spot_lattice = (
            self._setup_binomial_parameters()
        )  # (N,), (N,), (N+1,N+1)

        if self.binom_params.mc_paths is None:
            raise ValidationError("BinomialParams.mc_paths must be set for Asian binomial MC")

        mc_paths = int(self.binom_params.mc_paths)
        rng = np.random.default_rng(self.binom_params.random_seed)
        obs_idx = self._observation_indices(num_steps)  # (M,)

        K = self.valuation_ctx.strike

        # Vectorized simulation of binomial paths:
        # draw Bernoulli down-steps for each path and step
        downs = rng.random((mc_paths, num_steps)) > p[None, :]  # (I,N)
        # cumulative count of downs gives row index at each step
        down_counts = np.cumsum(downs, axis=1)  # (I,N)
        # prepend time 0 (row=0)
        row_idx = np.concatenate(
            [np.zeros((mc_paths, 1), dtype=int), down_counts], axis=1
        )  # (I,N+1)
        col_idx: np.ndarray = np.arange(num_steps + 1, dtype=int)  # (N+1,)

        # gather prices along each simulated path
        # spot_lattice is (N+1, N+1)
        # Advanced indexing: col_idx broadcasts to (I, N+1), so each
        # (row_idx[i, t], col_idx[t]) selects the node for path i at time t.
        prices = spot_lattice[row_idx, col_idx]  # (I, N+1)
        obs_prices = prices[:, obs_idx]  # (I, M)

        # Seasoned state: fold past observations into path averages
        n1 = self.spec.observed_count or 0
        s_bar = self.spec.observed_average

        if self.spec.averaging is AsianAveraging.GEOMETRIC:
            if np.any(obs_prices <= 0.0):
                raise NumericalError("Geometric averaging requires strictly positive path prices.")
            if n1 > 0 and s_bar is not None:
                n2 = obs_prices.shape[1]
                log_sum_future = np.sum(np.log(obs_prices), axis=1)
                avg_s = np.exp((n1 * np.log(s_bar) + log_sum_future) / (n1 + n2))  # (I,)
            else:
                avg_s = np.exp(np.mean(np.log(obs_prices), axis=1))  # (I,)
        else:
            if n1 > 0 and s_bar is not None:
                n2 = obs_prices.shape[1]
                sum_future = np.sum(obs_prices, axis=1)  # (I,)
                avg_s = (n1 * s_bar + sum_future) / (n1 + n2)  # (I,)
            else:
                avg_s = obs_prices.mean(axis=1)  # (I,)

        if self.spec.option_type is OptionType.CALL:
            payoffs = np.maximum(avg_s - K, 0.0)
        else:
            payoffs = np.maximum(K - avg_s, 0.0)

        return payoffs * float(np.prod(discount_factors))

    def _average_payoff(self, avg_price: np.ndarray | float) -> np.ndarray:
        """Compute Asian option intrinsic payoff given average price(s)."""
        K = self.valuation_ctx.strike
        if self.spec.option_type is OptionType.CALL:
            return np.maximum(avg_price - K, 0.0)
        return np.maximum(K - avg_price, 0.0)

    @staticmethod
    def _interp_value(x: float, grid: np.ndarray, values: np.ndarray) -> float:
        if grid[0] == grid[-1]:
            return float(values[0])
        return float(np.interp(x, grid, values))

    @staticmethod
    def _compute_ordering_bounds(
        lattice: np.ndarray,
        num_steps: int,
        observation_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute min/max running-average bounds achievable at each node.

        Works for any additive state variable: pass spot values for arithmetic
        averaging, or log-spot values for geometric averaging.

        For each node (row, t), avg_min is the mean obtained when all down moves
        precede all up moves, and avg_max is the mean when all up moves come first.

        Parameters
        ----------
        lattice : (N+1, N+1)
            Spot (or log-spot) lattice — lower-triangular.
        num_steps : int
            N, the number of tree steps.
        observation_indices : (M,)
            Tree-step indices at which fixing observations occur.

        Returns
        -------
        avg_min, avg_max : (N+1, N+1)
            Per-node min/max achievable running averages.
        """
        avg_min = np.zeros_like(lattice)  # (N+1, N+1)
        avg_max = np.zeros_like(lattice)  # (N+1, N+1)

        for t in range(num_steps + 1):
            obs = observation_indices[observation_indices <= t]
            if obs.size == 0:
                continue

            time_idx = np.arange(t + 1)
            row = time_idx[:, None]
            time = time_idx[None, :]

            downs_first_rows = np.minimum(time, row)
            prices_min = lattice[downs_first_rows, time]
            avg_min[: t + 1, t] = prices_min[:, obs].mean(axis=1)

            ups_first: np.ndarray = t - row
            ups_first_rows = np.maximum(0, time - ups_first)
            prices_max = lattice[ups_first_rows, time]
            avg_max[: t + 1, t] = prices_max[:, obs].mean(axis=1)

        return avg_min, avg_max

    @staticmethod
    def _apply_seasoned_bounds(
        *,
        avg_min: np.ndarray,
        avg_max: np.ndarray,
        observation_indices: np.ndarray,
        num_steps: int,
        n1: int,
        s_bar_state: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold seasoned observations into per-step min/max bounds.

        Parameters
        ----------
        avg_min, avg_max : (N+1, N+1)
            Fresh (future-only) ordering bounds from ``_compute_ordering_bounds``.
        observation_indices : (M,)
            Tree-step indices at which fixing observations occur.
        num_steps: int
            Number of binomial tree steps (N).
        n1 : int
            Number of already-observed fixings.
        s_bar_state : float
            Observed average (or log-average for geometric).

        Returns
        -------
        avg_min_out, avg_max_out : (N+1, N+1)
            Bounds adjusted for the seasoned portion.
        """

        if n1 <= 0:
            raise ValidationError("Seasoned bounds require positive observed_count")

        n_future: np.ndarray = np.searchsorted(  # (N+1,)
            observation_indices,
            np.arange(num_steps + 1),
            side="right",
        ).astype(float)
        n_total: np.ndarray = n1 + n_future  # (N+1,)

        avg_min_out = (n1 * s_bar_state + n_future[None, :] * avg_min) / n_total[
            None, :
        ]  # (N+1, N+1)
        avg_max_out = (n1 * s_bar_state + n_future[None, :] * avg_max) / n_total[
            None, :
        ]  # (N+1, N+1)
        return avg_min_out, avg_max_out

    @staticmethod
    def _update_child_averages(
        *,
        grid_here: np.ndarray,
        s_up: np.ndarray,
        s_down: np.ndarray,
        observation_indices: np.ndarray,
        t: int,
        n1: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update running-average state when advancing one tree step.

        Parameters
        ----------
        grid_here : (k, t+1)
            Representative averages at the current time step. Each column corresponds to a node; the k
            linearly spaced values from S_avg,min to S_avg,max at that node.
        s_up, s_down : (t+1,)
            Spot (or log-spot) at the up/down child nodes.
        observation_indices : (M,)
            Fixing-date tree indices.
        t : int
            Current tree step.
        n1 : int
            Number of past (seasoned) observations.

        Returns
        -------
        avg_up, avg_down : (k, t+1)
            Updated running averages after incorporating the child spot.
        """
        obs_count_now = int(np.searchsorted(observation_indices, t, side="right"))
        obs_so_far = n1 + obs_count_now
        obs_count_next = int(np.searchsorted(observation_indices, t + 1, side="right"))
        new_obs = obs_count_next - obs_count_now
        if new_obs > 0:
            avg_up = (obs_so_far * grid_here + new_obs * s_up) / (obs_so_far + new_obs)  # (k, t+1)
            avg_down = (obs_so_far * grid_here + new_obs * s_down) / (
                obs_so_far + new_obs
            )  # (k, t+1)
            return avg_up, avg_down
        return grid_here, grid_here

    @staticmethod
    def _interp_child_values(
        *,
        avg_up: np.ndarray,
        avg_down: np.ndarray,
        avg_grid: np.ndarray,
        values: np.ndarray,
        rows: np.ndarray,
        t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate continuation values from child average grids.

        Parameters
        ----------
        avg_up, avg_down : (k, t+1)
            Running averages at the up/down child nodes.
        avg_grid : (k, N+1, N+1)
            Full representative-average grid (for spot prices).
        values : (k, N+1, N+1)
            Full option-value grid.
        rows : (t+1,)
            Row indices of active nodes at step *t*.

        Returns
        -------
        v_up, v_down : (k, t+1)
            Interpolated option values at the child average positions.
        """
        v_up = np.empty_like(avg_up)  # (k, t+1)
        v_down = np.empty_like(avg_down)  # (k, t+1)
        for j, row_idx in enumerate(rows):
            row = int(row_idx)
            v_up[:, j] = np.interp(avg_up[:, j], avg_grid[:, row, t + 1], values[:, row, t + 1])
            v_down[:, j] = np.interp(
                avg_down[:, j], avg_grid[:, row + 1, t + 1], values[:, row + 1, t + 1]
            )
        return v_up, v_down

    def _solve_hull(self) -> tuple[np.ndarray, np.ndarray]:
        """Price an Asian option using Hull's representative-averages binomial tree.

        At each node, *k* linearly spaced average values between the attainable
        minimum and maximum are maintained. Backward induction interpolates
        continuation values from the child nodes' grids. For geometric
        averaging, all state variables are stored in log space.

        For seasoned Asians (``spec.observed_average`` and
        ``spec.observed_count`` set), past observations are folded into the
        running average at every node.  At tree step *t* the denominator
        becomes ``n₁ + t + 1`` (total observations so far) instead of
        ``t + 1``.

        Notation
        --------
        N = num_steps, k = asian_tree_averages, M = len(observation_indices).

        Core arrays
        ~~~~~~~~~~~
        spot_lattice      (N+1, N+1)    CRR spot prices (lower-triangular)
        avg_min, avg_max   (N+1, N+1)    min/max achievable running average per node
        avg_grid           (k, N+1, N+1) representative averages per node
        values             (k, N+1, N+1) option values per average bucket

        Per backward step *t*  (loop-local)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rows               (t+1,)        row indices of active nodes
        s_up, s_down       (t+1,)        spot (or log-spot) at up/down children
        grid_here          (k, t+1)      representative averages at step t
        avg_up, avg_down   (k, t+1)      updated running averages at children
        v_up, v_down       (k, t+1)      interpolated option values at children
        continuation       (k, t+1)      discounted expected continuation value

        Returns
        -------
        tuple of (values, avg_grid)
            values : (k, N+1, N+1) — option values per average bucket
            avg_grid : (k, N+1, N+1) — representative averages
        """
        num_steps = int(self.binom_params.num_steps)
        logger.debug(
            "Binomial Asian Hull num_steps=%d averages=%s",
            num_steps,
            self.binom_params.asian_tree_averages,
        )
        discount_factors, p, spot_lattice = (
            self._setup_binomial_parameters()
        )  # (N,), (N,), (N+1,N+1)
        observation_indices = self._observation_indices(num_steps)  # (M,)

        averaging = self.spec.averaging
        if averaging not in (AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC):
            raise ValidationError("Unsupported Asian averaging type for Hull binomial valuation.")

        if self.binom_params.asian_tree_averages is None:
            raise ValidationError(
                "BinomialParams.asian_tree_averages must be set for Hull binomial Asian valuation"
            )

        # Seasoned state: n₁ past observations with average S̄
        n1 = self.spec.observed_count or 0
        s_bar = self.spec.observed_average

        k = int(self.binom_params.asian_tree_averages)
        if averaging is AsianAveraging.GEOMETRIC:
            if np.any(spot_lattice <= 0.0):
                raise NumericalError(
                    "Hull binomial geometric averaging requires strictly positive spot lattice."
                )
            log_spot_lattice = np.log(spot_lattice)  # (N+1, N+1)
            avg_min, avg_max = self._compute_ordering_bounds(  # (N+1, N+1) each
                log_spot_lattice,
                num_steps,
                observation_indices,
            )
            s_bar_state = np.log(s_bar) if s_bar is not None else 0.0
        else:
            avg_min, avg_max = self._compute_ordering_bounds(  # (N+1, N+1) each
                spot_lattice,
                num_steps,
                observation_indices,
            )
            s_bar_state = s_bar if s_bar is not None else 0.0

        # Fold past observations into the bounds.
        # Fresh bounds are averages over t+1 future prices; seasoned bounds
        # are averages over n₁ + t + 1 total observations.
        if n1 > 0 and s_bar is not None:
            avg_min, avg_max = self._apply_seasoned_bounds(
                avg_min=avg_min,
                avg_max=avg_max,
                observation_indices=observation_indices,
                num_steps=num_steps,
                n1=n1,
                s_bar_state=s_bar_state,
            )

        avg_grid: np.ndarray = np.zeros(
            (k, num_steps + 1, num_steps + 1), dtype=float
        )  # (k, N+1, N+1)
        for t in range(num_steps + 1):
            rows = np.arange(t + 1)
            alpha = np.linspace(0.0, 1.0, k)[:, None]
            avg_grid[:, rows, t] = (
                avg_min[rows, t][None, :] + (avg_max[rows, t] - avg_min[rows, t])[None, :] * alpha
            )

        values = np.zeros_like(avg_grid)  # (k, N+1, N+1)

        for row in range(num_steps + 1):
            maturity_avg = avg_grid[:, row, num_steps]
            if averaging is AsianAveraging.GEOMETRIC:
                maturity_avg = np.exp(maturity_avg)
            values[:, row, num_steps] = self._average_payoff(maturity_avg)

        # values[0,:,:] are option values corresponding to S_avg,min.
        # values[-1,:,:] are option values corresponding to S_avg,max.
        is_american = self.valuation_ctx.spec.exercise_type is ExerciseType.AMERICAN

        for t in range(num_steps - 1, -1, -1):
            rows = np.arange(t + 1)  # (t+1,)
            if averaging is AsianAveraging.GEOMETRIC:
                s_up = log_spot_lattice[rows, t + 1]  # (t+1,)
                s_down = log_spot_lattice[rows + 1, t + 1]  # (t+1,)
            else:
                s_up = spot_lattice[rows, t + 1]  # (t+1,)
                s_down = spot_lattice[rows + 1, t + 1]  # (t+1,)
            grid_here = avg_grid[:, rows, t]  # (k, t+1)

            avg_up, avg_down = self._update_child_averages(
                grid_here=grid_here,
                s_up=s_up,
                s_down=s_down,
                observation_indices=observation_indices,
                t=t,
                n1=n1,
            )
            v_up, v_down = self._interp_child_values(
                avg_up=avg_up,
                avg_down=avg_down,
                avg_grid=avg_grid,
                values=values,
                rows=rows,
                t=t,
            )

            continuation = discount_factors[t] * (p[t] * v_up + (1.0 - p[t]) * v_down)  # (k, t+1)

            if is_american:
                # Early exercise is only meaningful once at least one fixing
                # observation has been recorded; before that the running average
                # is undefined (avg_grid is zero-initialised) and would produce
                # a spurious intrinsic value.
                obs_count = int(np.searchsorted(observation_indices, t, side="right"))
                total_obs = n1 + obs_count
                if total_obs > 0:
                    exercise_avg = grid_here
                    if averaging is AsianAveraging.GEOMETRIC:
                        exercise_avg = np.exp(exercise_avg)
                    exercise = self._average_payoff(exercise_avg)
                    values[:, rows, t] = np.maximum(continuation, exercise)
                else:
                    values[:, rows, t] = continuation
            else:
                values[:, rows, t] = continuation

        return values, avg_grid

    def present_value(self) -> float:
        """Return present value for Asian option under selected binomial mode."""
        with log_timing(logger, "Binomial Asian present_value", self.binom_params.log_timings):
            if self.binom_params.mc_paths is not None:
                pv_pathwise = self._solve_mc()
                return float(np.mean(pv_pathwise))

            values, _ = self._solve_hull()
            return float(values[0, 0, 0])

    def solve(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Solve the Asian binomial problem.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            Pathwise discounted payoffs for MC mode, otherwise
            ``(values, average_grid)`` for Hull representative-average mode.
        """
        if self.binom_params.mc_paths is not None:
            return self._solve_mc()
        return self._solve_hull()
