"""Top-level exports for common derivatives_pricing types."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("derivatives-pricing")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .exceptions import (
    DerivativesPricingError,
    ValidationError,
    ConfigurationError,
    UnsupportedFeatureError,
    NumericalError,
    ArbitrageViolationError,
    ConvergenceError,
    StabilityError,
)
from .market_environment import MarketData, CorrelationContext
from .rates import DiscountCurve
from .stochastic_processes import (
    SimulationConfig,
    GBMParams,
    JDParams,
    SRDParams,
    GBMProcess,
    JDProcess,
    SRDProcess,
    PathSimulation,
)
from .enums import (
    OptionType,
    ExerciseType,
    PricingMethod,
    PDEMethod,
    PDESpaceGrid,
    PDEEarlyExercise,
    DayCountConvention,
    AsianAveraging,
    GreekCalculationMethod,
    ImpliedVolMethod,
    BarrierDirection,
    BarrierAction,
    BarrierMonitoring,
    RebateTiming,
)
from .valuation import (
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
    as_underlying_data,
    PayoffSpec,
    PayoffBoundaryModel,
    WingBoundary,
    AsianSpec,
    BarrierSpec,
    MonteCarloParams,
    BinomialParams,
    PDEParams,
)
from .valuation.implied_volatility import ImpliedVolResult, implied_volatility

__all__ = [
    # Market data
    "MarketData",
    "CorrelationContext",
    "DiscountCurve",
    # Stochastic processes
    "SimulationConfig",
    "GBMParams",
    "JDParams",
    "SRDParams",
    "GBMProcess",
    "JDProcess",
    "SRDProcess",
    "PathSimulation",
    # Enums
    "OptionType",
    "ExerciseType",
    "PricingMethod",
    "PDEMethod",
    "PDESpaceGrid",
    "PDEEarlyExercise",
    "DayCountConvention",
    "AsianAveraging",
    "GreekCalculationMethod",
    "ImpliedVolMethod",
    "BarrierDirection",
    "BarrierAction",
    "BarrierMonitoring",
    "RebateTiming",
    # Valuation
    "VanillaSpec",
    "OptionValuation",
    "UnderlyingData",
    "as_underlying_data",
    "PayoffSpec",
    "PayoffBoundaryModel",
    "WingBoundary",
    "AsianSpec",
    "BarrierSpec",
    "MonteCarloParams",
    "BinomialParams",
    "PDEParams",
    "ImpliedVolResult",
    "implied_volatility",
    # Exceptions
    "DerivativesPricingError",
    "ValidationError",
    "ConfigurationError",
    "UnsupportedFeatureError",
    "NumericalError",
    "ArbitrageViolationError",
    "ConvergenceError",
    "StabilityError",
]
