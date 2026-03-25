"""Contract specification dataclasses used by valuation engines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import datetime as dt

import numpy as np

from ..enums import AsianAveraging, ExerciseType, OptionType
from ..exceptions import ConfigurationError, ValidationError
from ..utils import validate_naive_datetime


@dataclass(frozen=True, slots=True)
class VanillaSpec:
    """Contract specification for a vanilla option.

    Parameters
    ----------
    option_type
        Vanilla option direction (CALL or PUT).
    exercise_type
        Exercise style (EUROPEAN or AMERICAN).
    strike
        Strike price.
    maturity
        Contract maturity datetime.
    currency
        Optional contract currency. If ``None``, the underlying currency is used for valuation.
    contract_size
        Contract multiplier (default 100).  Not applied by ``OptionValuation``
        (which returns per-unit values); intended for portfolio-level position
        sizing (e.g. position delta = unit delta × contract_size).
    """

    option_type: OptionType  # CALL / PUT
    exercise_type: ExerciseType  # EUROPEAN / AMERICAN
    strike: float
    maturity: dt.datetime
    currency: str | None = None
    contract_size: int | float = 100

    def __post_init__(self) -> None:
        """Validate option_type/exercise_type and coerce strike."""
        validate_naive_datetime(
            self.maturity,
            "maturity",
            type_error_cls=ConfigurationError,
        )

        if not isinstance(self.option_type, OptionType):
            raise ConfigurationError(
                f"option_type must be OptionType enum, got {type(self.option_type).__name__}"
            )
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValidationError(
                "VanillaSpec.option_type must be OptionType.CALL or OptionType.PUT"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if self.strike is None:
            raise ValidationError("VanillaSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("VanillaSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValidationError("VanillaSpec.strike must be finite")
        if strike < 0.0:
            raise ValidationError("VanillaSpec.strike must be >= 0")
        object.__setattr__(self, "strike", strike)


@dataclass(frozen=True, slots=True)
class WingBoundary:
    """Affine boundary model on one wing: ``payoff(S) ~ slope * S + intercept``.

    The PDE engine uses this as an affine approximation at the truncated
    computational boundary.  On spot grids this will often coincide with
    the true payoff tail behaviour; on log-spot grids it should be
    interpreted as a local boundary model at the finite boundary.
    """

    slope: float
    intercept: float


@dataclass(frozen=True, slots=True)
class PayoffBoundaryModel:
    """Left/right affine boundary models for a custom payoff.

    Used by the PDE solver to set continuation values at the truncated domain
    boundaries. On spot grids these often coincide with true payoff tail
    behaviour; on log-spot grids they are best interpreted as affine
    approximations at the finite boundaries used by the PDE grid.

    If not supplied on ``PayoffSpec``, the PDE engine will fit them
    numerically from the payoff callable.
    """

    left: WingBoundary
    right: WingBoundary


@dataclass(frozen=True, slots=True)
class PayoffSpec:
    """Contract specification for a single-contract custom payoff.

    This is useful for pricing payoffs that are not representable as a single vanilla
    call/put (e.g., capped combinations), while still treating the product as ONE
    contract for exercise decisions (American pricing compares intrinsic vs continuation
    on the full payoff).

    Parameters
    ----------
    exercise_type
        Exercise style (EUROPEAN or AMERICAN).
    maturity
        Contract maturity datetime.
    payoff_fn
        Vectorized payoff callable in spot, accepting ``float | np.ndarray`` and
        returning ``np.ndarray``.
    currency
        Optional contract currency. If ``None``, the underlying currency is used.
    contract_size
        Contract multiplier (default 100).  Not applied by ``OptionValuation``
        (which returns per-unit values); intended for portfolio-level position
        sizing.
    boundary_model
        Optional explicit affine boundary models for PDE boundary conditions.
        If ``None``, the PDE solver will fit them numerically from ``payoff_fn``
        using the actual truncated PDE boundary neighborhoods.

    Notes
    -----
    ``strike`` is intentionally fixed to ``None`` for interface compatibility with
    ``OptionValuation``.
    """

    exercise_type: ExerciseType
    maturity: dt.datetime
    payoff_fn: Callable[[np.ndarray | float], np.ndarray]
    currency: str | None = None
    contract_size: int | float = 100
    boundary_model: PayoffBoundaryModel | None = None

    # Kept for compatibility with vanilla valuation interfaces
    strike: None = None

    def __post_init__(self) -> None:
        validate_naive_datetime(
            self.maturity,
            "maturity",
            type_error_cls=ConfigurationError,
        )

        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )
        if not callable(self.payoff_fn):
            raise ConfigurationError("payoff_fn must be callable")
        if self.boundary_model is not None and not isinstance(
            self.boundary_model, PayoffBoundaryModel
        ):
            raise ConfigurationError(
                "boundary_model must be PayoffBoundaryModel or None, "
                f"got {type(self.boundary_model).__name__}"
            )

    def payoff(self, spot: np.ndarray | float) -> np.ndarray:
        """Vectorized payoff as a function of spot."""
        # Ensure a float ndarray output (for downstream math and boolean comparisons).
        return np.asarray(self.payoff_fn(spot), dtype=float)


@dataclass(frozen=True, slots=True)
class AsianSpec:
    """Contract specification for an Asian option.

    Asian options are path-dependent options where the payoff depends on the average
    price of the underlying over a specified averaging period.

    Parameters
    ----------
    averaging : AsianAveraging
        AsianAveraging.ARITHMETIC or AsianAveraging.GEOMETRIC
    option_type : OptionType
        OptionType.CALL or OptionType.PUT to specify payoff direction
    exercise_type : ExerciseType
        Exercise style (EUROPEAN or AMERICAN)
    strike : float
        Strike price
    maturity : dt.datetime
        Option maturity date
    currency : str, optional
        Currency denomination
    averaging_start : dt.datetime, optional
        Start of averaging period. If None, uses pricing date.
    num_observations : int, optional
        Number of **future** equally spaced averaging observation time points
        within the averaging window.  For fresh (unseasoned) options this equals
        the total observation count.  For seasoned options, set this to the
        number of *remaining* observations; ``observed_count`` tracks the
        already-observed fixings separately.
    contract_size : int | float
        Contract multiplier (default 100).  Not applied by ``OptionValuation``
        (which returns per-unit values); intended for portfolio-level position
        sizing (e.g. position delta = unit delta × contract_size).
    fixing_dates : Sequence[dt.datetime], optional
        Explicit fixing (observation) dates for discrete averaging.
        When provided, only the spot
        prices on these dates contribute to the average — any other grid dates
        (pricing date, ex-dividend dates, maturity) are simulated but excluded
        from the average.  Dates must be in ascending order and fall within
        ``[averaging_start (or pricing_date), maturity]``.  Mutually exclusive
        with ``num_observations``.
    observed_average : float, optional
        For seasoned Asians: the realised average price over the already-observed
        period.  Must be provided together with ``observed_count``.
    observed_count : int, optional
        For seasoned Asians: the number of already-observed fixings (n₁).
        Must be provided together with ``observed_average``.

    Notes
    -----
    - Arithmetic average: S_avg = (1/N) * Σ S_i
    - Geometric average: S_avg = (Π S_i)^(1/N)
    - Payoff for call: max(S_avg - K, 0)
    - Payoff for put: max(K - S_avg, 0)
    - European and American exercise are supported depending on pricing method
    """

    averaging: AsianAveraging
    option_type: OptionType  # CALL or PUT
    exercise_type: ExerciseType  # EUROPEAN or AMERICAN
    strike: float
    maturity: dt.datetime
    currency: str | None = None
    averaging_start: dt.datetime | None = None
    num_observations: int | None = None
    contract_size: int | float = 100
    fixing_dates: Sequence[dt.datetime] | None = None
    observed_average: float | None = None
    observed_count: int | None = None

    def __post_init__(self) -> None:
        """Validate Asian option specification."""
        validate_naive_datetime(
            self.maturity,
            "maturity",
            type_error_cls=ConfigurationError,
        )
        if self.averaging_start is not None:
            validate_naive_datetime(
                self.averaging_start,
                "averaging_start",
                type_error_cls=ConfigurationError,
            )

        if not isinstance(self.averaging, AsianAveraging):
            raise ConfigurationError(
                f"averaging must be AsianAveraging enum, got {type(self.averaging).__name__}"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if not isinstance(self.option_type, OptionType):
            raise ConfigurationError(
                f"option_type must be OptionType enum, got {type(self.option_type).__name__}"
            )
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValidationError("AsianSpec.option_type must be OptionType.CALL or OptionType.PUT")

        if self.strike is None:
            raise ValidationError("AsianSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("AsianSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValidationError("AsianSpec.strike must be finite")
        if strike < 0.0:
            raise ValidationError("AsianSpec.strike must be >= 0")
        object.__setattr__(self, "strike", strike)

        # Exactly one schedule source is required.
        if (self.fixing_dates is None) == (self.num_observations is None):
            raise ValidationError(
                "AsianSpec requires exactly one of fixing_dates or num_observations."
            )

        if self.averaging_start is not None and self.averaging_start > self.maturity:
            raise ValidationError("averaging_start must be on or before maturity.")

        if self.num_observations is not None:
            if not isinstance(self.num_observations, int) or self.num_observations < 2:
                raise ValidationError("num_observations must be an integer >= 2")

        # fixing_dates: coerce to tuple, validate ordering and bounds
        if self.fixing_dates is not None:
            dates = tuple(self.fixing_dates)
            if not dates:
                raise ValidationError("fixing_dates must be non-empty when provided.")
            for d in dates:
                validate_naive_datetime(
                    d,
                    "fixing_dates entry",
                    type_error_cls=ConfigurationError,
                )
            if len(dates) != len(set(dates)):
                raise ValidationError("fixing_dates must contain unique dates.")
            if any(dates[i] >= dates[i + 1] for i in range(len(dates) - 1)):
                raise ValidationError("fixing_dates must be in strictly ascending order.")
            # Bounds are checked later against the pricing date (not known here);
            # maturity is available so we can at least ensure dates don't exceed it.
            if dates[-1] > self.maturity:
                raise ValidationError("fixing_dates must not extend beyond maturity.")
            if self.averaging_start is not None:
                raise ValidationError("if fixing_dates are provided, averaging_start must be None")
            object.__setattr__(self, "fixing_dates", dates)

        # Seasoned Asian: observed_average and observed_count must be both set or both None
        if (self.observed_average is None) != (self.observed_count is None):
            raise ValidationError(
                "observed_average and observed_count must both be provided or both omitted."
            )
        if self.observed_average is not None:
            try:
                obs_avg = float(self.observed_average)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("observed_average must be numeric") from exc
            if not np.isfinite(obs_avg):
                raise ValidationError("observed_average must be finite")
            if obs_avg <= 0.0:
                raise ValidationError("observed_average must be > 0")
            object.__setattr__(self, "observed_average", obs_avg)

            if not isinstance(self.observed_count, int) or self.observed_count < 1:
                raise ValidationError("observed_count must be a positive integer")
