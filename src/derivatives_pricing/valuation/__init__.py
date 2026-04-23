"""Option valuation and pricing engines.

This module provides a unified interface for pricing vanilla and custom options
using various methods: Monte Carlo simulation, Binomial trees, Black-Scholes-Merton
analytical formulas, and PDE finite difference methods.

Public API
----------
Core classes:
    OptionValuation: Main dispatcher for option pricing
    VanillaSpec: Contract specification for vanilla options
    PayoffSpec: Contract specification for custom payoffs
    UnderlyingData: Minimal underlying data container

Parameter classes:
    MonteCarloParams: Configuration for Monte Carlo pricing
    BinomialParams: Configuration for Binomial tree pricing
    PDEParams: Configuration for PDE finite difference pricing
    ValuationParams: Union type for all parameter classes
"""

from .contracts import (
    VanillaSpec,
    PayoffSpec,
    AsianSpec,
    BarrierSpec,
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
