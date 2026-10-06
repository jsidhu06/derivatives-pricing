"""Enums for option valuation."""

from enum import Enum

__all__ = [
    "OptionType",
    "AsianAveraging",
    "ExerciseType",
    "PricingMethod",
    "GreekCalculationMethod",
    "ImpliedVolMethod",
    "PDEMethod",
    "PDESpaceGrid",
    "PDEEarlyExercise",
    "DayCountConvention",
    "BarrierDirection",
    "BarrierAction",
    "BarrierMonitoring",
    "RebateTiming",
]


class OptionType(Enum):
    """Call or put option type."""

    CALL = "call"
    PUT = "put"


class AsianAveraging(Enum):
    """Averaging method for Asian option payoffs."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class ExerciseType(Enum):
    """European (expiry only) or American (any time) exercise."""

    EUROPEAN = "european"
    AMERICAN = "american"


class PricingMethod(Enum):
    """Numerical method used by ``OptionValuation`` to price an option.

    ``ANALYTICAL`` is closed-form pricing under GBM dynamics:
    Black-Scholes-Merton for vanillas, Reiner-Rubinstein (with the
    Broadie-Glasserman-Kou discrete correction) for barriers, and
    Kemna-Vorst / Turnbull-Wakeman for Asians.
    """

    ANALYTICAL = "analytical"
    BINOMIAL = "binomial"
    PDE_FD = "pde_fd"
    MONTE_CARLO = "monte_carlo"
    # Deprecated alias of ANALYTICAL (the pre-0.3 name).
    BSM = "analytical"

    @classmethod
    def _missing_(cls, value: object) -> "PricingMethod | None":
        # Accept the pre-0.3 value "bsm" for ANALYTICAL.
        return cls.ANALYTICAL if value == "bsm" else None


class GreekCalculationMethod(Enum):
    """Algorithm used to compute Greeks on an ``OptionValuation``.

    Not every method is available for every pricing engine — see the
    Greek method docstrings on ``OptionValuation`` for the compatibility
    matrix.
    """

    ANALYTICAL = "analytical"
    TREE = "tree"
    GRID = "grid"
    PATHWISE = "pathwise"
    LIKELIHOOD_RATIO = "likelihood_ratio"
    NUMERICAL = "numerical"


class ImpliedVolMethod(Enum):
    """Root-finding algorithm for the ``implied_volatility`` solver."""

    NEWTON_RAPHSON = "newton_raphson"
    BISECTION = "bisection"
    BRENTQ = "brentq"


class PDEMethod(Enum):
    """Finite-difference time-stepping scheme for the PDE solver."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    EXPLICIT_HULL = "explicit_hull"
    CRANK_NICOLSON = "crank_nicolson"


class PDESpaceGrid(Enum):
    """Spatial variable for the PDE finite-difference grid."""

    SPOT = "spot"
    LOG_SPOT = "log_spot"


class PDEEarlyExercise(Enum):
    """Early-exercise enforcement algorithm for American PDE pricing."""

    INTRINSIC = "intrinsic"
    GAUSS_SEIDEL = "gauss_seidel"


class DayCountConvention(Enum):
    """Day-count convention for year-fraction calculations."""

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    ACT_365_25 = "ACT/365.25"
    THIRTY_360_US = "30/360 US"


class BarrierDirection(Enum):
    """Direction the underlying must move to trigger the barrier."""

    UP = "up"
    DOWN = "down"


class BarrierAction(Enum):
    """Whether the barrier knocks the option in or out."""

    IN = "in"
    OUT = "out"


class BarrierMonitoring(Enum):
    """How frequently the barrier is monitored."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class RebateTiming(Enum):
    """When the rebate is paid upon barrier trigger."""

    AT_HIT = "at_hit"
    AT_EXPIRY = "at_expiry"
