"""Market data containers for valuation and simulation."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
import datetime as dt
import numpy as np
from .enums import DayCountConvention
from .rates import DiscountCurve
from .exceptions import ValidationError
from .utils import validate_naive_datetime


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market data required for valuation/simulation."""

    pricing_date: dt.datetime
    discount_curve: DiscountCurve
    currency: str
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F

    def __post_init__(self) -> None:
        validate_naive_datetime(self.pricing_date, "pricing_date")
        if not isinstance(self.discount_curve, DiscountCurve):
            raise ValidationError(
                f"discount_curve must be a DiscountCurve, got {type(self.discount_curve).__name__}"
            )
        if not isinstance(self.currency, str) or not self.currency:
            raise ValidationError("currency must be a non-empty string")
        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be a DayCountConvention enum, "
                f"got {type(self.day_count_convention).__name__}"
            )


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Shared correlation/scenario context for multi-asset simulation.

    Users supply a symmetric correlation matrix (with ones on the diagonal),
    an ordered sequence of asset names and the shared simulation time grid.
    The Cholesky factor and the name-to-index mapping are derived in
    ``__post_init__``.

    The time grid is part of the scenario: column ``k`` of
    ``random_numbers`` drives the interval ``(time_grid[k], time_grid[k+1])``
    for every asset, so every process in the context must simulate on
    exactly this grid. It must therefore already contain every asset's
    ex-dividend dates and every contract's event dates (maturities, barrier
    monitoring and Asian fixing dates); nothing is merged in later. Build it
    with :func:`shared_time_grid`, passing every contract priced on the
    context as ``specs`` and every asset's parameters as ``process_params``.

    Parameters
    ----------
    correlation_matrix : np.ndarray, shape (n_assets, n_assets)
        Symmetric, positive-definite correlation matrix with ones on
        the diagonal.  Row/column *i* corresponds to ``asset_names[i]``.
    random_numbers : np.ndarray, shape (n_assets, n_time_intervals, n_paths)
        Pre-generated independent standard-normal draws.  Axis 0 is
        indexed by ``asset_names``; axis 1 must equal
        ``len(time_grid) - 1``. See :meth:`from_seed` to draw them.
    asset_names : Sequence[str]
        Ordered identifiers for each asset/process.  Length must equal
        ``correlation_matrix.shape[0]``.
    time_grid : Sequence[datetime]
        Shared simulation dates, strictly ascending, starting at the pricing
        date of every process in the context.
    """

    correlation_matrix: np.ndarray
    random_numbers: np.ndarray
    asset_names: Sequence[str]
    time_grid: Sequence[dt.datetime]

    # Derived (not user-facing) -------------------------------------------------
    cholesky_matrix: np.ndarray = field(init=False, repr=False)
    _asset_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        C = np.asarray(self.correlation_matrix, dtype=float)

        # --- shape checks ---
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValidationError(f"correlation_matrix must be square, got shape {C.shape}")
        n = C.shape[0]

        if len(self.asset_names) != n:
            raise ValidationError(
                f"asset_names length ({len(self.asset_names)}) must match "
                f"correlation_matrix dimension ({n})"
            )

        # --- duplicate names ---
        if len(set(self.asset_names)) != len(self.asset_names):
            raise ValidationError("asset_names must contain unique entries")

        # --- symmetry ---
        if not np.allclose(C, C.T, atol=1e-12):
            raise ValidationError("correlation_matrix must be symmetric")

        # --- unit diagonal ---
        if not np.allclose(np.diag(C), 1.0, atol=1e-12):
            raise ValidationError(
                "correlation_matrix diagonal must be 1 (it is a correlation matrix, not covariance)"
            )

        # --- positive-definite (Cholesky will fail otherwise) ---
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("correlation_matrix is not positive-definite") from exc

        object.__setattr__(self, "cholesky_matrix", L)

        # --- random_numbers shape ---
        rn = self.random_numbers
        if rn.ndim != 3:
            raise ValidationError(
                f"random_numbers must be 3-D (n_assets, n_steps, n_paths), got ndim={rn.ndim}"
            )
        if rn.shape[0] != n:
            raise ValidationError(
                f"random_numbers axis-0 ({rn.shape[0]}) must match number of assets ({n})"
            )

        # --- shared time grid ---
        grid = tuple(self.time_grid)
        if len(grid) < 2:
            raise ValidationError(f"time_grid must contain at least two dates, got {len(grid)}")
        for grid_date in grid:
            validate_naive_datetime(grid_date, "time_grid entry")
        for earlier, later in zip(grid[:-1], grid[1:]):
            if not later > earlier:
                raise ValidationError(
                    f"time_grid must be strictly ascending, got {later} after {earlier}"
                )
        object.__setattr__(self, "time_grid", grid)
        if rn.shape[1] != len(grid) - 1:
            raise ValidationError(
                f"random_numbers axis-1 ({rn.shape[1]}) must equal the number of "
                f"time_grid intervals ({len(grid) - 1})"
            )

        # --- build index mapping ---
        object.__setattr__(
            self, "_asset_index", {name: i for i, name in enumerate(self.asset_names)}
        )

    @classmethod
    def from_seed(
        cls,
        *,
        correlation_matrix: np.ndarray,
        asset_names: Sequence[str],
        time_grid: Sequence[dt.datetime],
        paths: int,
        random_seed: int | None = None,
    ) -> CorrelationContext:
        """Build a context whose independent normals are drawn here.

        The draws have shape ``(n_assets, len(time_grid) - 1, paths)``, so
        they match the grid by construction.
        """
        if int(paths) <= 0:
            raise ValidationError(f"paths must be a positive integer, got {paths}")
        rng = np.random.default_rng(random_seed)
        random_numbers = rng.standard_normal((len(asset_names), len(time_grid) - 1, int(paths)))
        return cls(
            correlation_matrix=correlation_matrix,
            random_numbers=random_numbers,
            asset_names=asset_names,
            time_grid=time_grid,
        )

    @property
    def paths(self) -> int:
        """Number of simulated paths the draws cover."""
        return int(self.random_numbers.shape[2])

    def asset_index(self, name: str) -> int:
        """Return the integer index for *name*.

        Raises ``ValidationError`` if *name* is not in ``asset_names``.
        """
        try:
            return self._asset_index[name]
        except KeyError:
            raise ValidationError(
                f"'{name}' not found in asset_names: {list(self.asset_names)}"
            ) from None
