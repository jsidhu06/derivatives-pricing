"""Option valuation and pricing engines.

This module provides a unified interface for pricing vanilla, custom-payoff,
Asian, and barrier options using various methods: Monte Carlo simulation,
Binomial trees, Black-Scholes-Merton analytical formulas, and PDE finite
difference methods.

Public API
----------
Core classes:
    OptionValuation: Main dispatcher for option pricing
    VanillaSpec: Contract specification for vanilla options
    PayoffSpec: Contract specification for custom payoffs
    AsianSpec: Contract specification for Asian options
    BarrierSpec: Contract specification for barrier options
    OptionSpec: Type alias — union of the four spec classes above
    UnderlyingData: Minimal underlying data container

Barrier enums:
    BarrierDirection: UP / DOWN
    BarrierAction: IN / OUT
    BarrierMonitoring: CONTINUOUS / DISCRETE
    RebateTiming: AT_HIT / AT_EXPIRY

Parameter classes:
    MonteCarloParams: Configuration for Monte Carlo pricing
    BinomialParams: Configuration for Binomial tree pricing
    PDEParams: Configuration for PDE finite difference pricing
    ValuationParams: Union type for all parameter classes
"""

from ..enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    RebateTiming,
)
from .contracts import (
    VanillaSpec,
    PayoffSpec,
    AsianSpec,
    BarrierSpec,
    OptionSpec,
    WingBoundary,
    PayoffBoundaryModel,
)
from .core import OptionValuation, UnderlyingData, as_underlying_data
from .params import (
    MonteCarloParams,
    BinomialParams,
    PDEParams,
    ValuationParams,
)
from .implied_volatility import ImpliedVolResult, implied_volatility

__all__ = [
    # Option contract classes
    "VanillaSpec",
    "PayoffSpec",
    "PayoffBoundaryModel",
    "WingBoundary",
    "AsianSpec",
    "BarrierSpec",
    "OptionSpec",
    # Barrier enums
    "BarrierDirection",
    "BarrierAction",
    "BarrierMonitoring",
    "RebateTiming",
    # Core valuation classes
    "OptionValuation",
    "UnderlyingData",
    "as_underlying_data",
    # Parameter classes
    "MonteCarloParams",
    "BinomialParams",
    "PDEParams",
    "ValuationParams",
    "ImpliedVolResult",
    "implied_volatility",
]
