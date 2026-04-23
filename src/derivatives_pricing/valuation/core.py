"""Core valuation contracts and dispatcher.

This module is the central orchestration layer for pricing:

- Spec dataclasses (`VanillaSpec`, `PayoffSpec`, `AsianSpec`)
- Underlying data container (`UnderlyingData`)
- Registry-based dispatcher (`OptionValuation`) that maps
    `(PricingMethod, ExerciseType)` to a private implementation engine

Design notes
------------
- Deterministic methods (`BSM`, `BINOMIAL`, `PDE_FD`) operate on
    `UnderlyingData`.
- Monte Carlo methods operate on `PathSimulation` instances.
- Greeks default to engine-native methods when available, otherwise
    fall back to bump-and-revalue via `_bump_underlying(...)` which
    constructs a fresh underlying with the bumped parameter.
"""

from __future__ import annotations
from dataclasses import dataclass, replace as dc_replace
from collections.abc import Sequence
from functools import wraps
from typing import Any
import datetime as dt
import logging
import threading
import numpy as np
import pandas as pd
from ..utils import calculate_year_fraction
from ..stochastic_processes import PathSimulation, GBMProcess
from ..exceptions import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..enums import (
    AsianAveraging,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    OptionType,
    ExerciseType,
    PricingMethod,
    GreekCalculationMethod,
)
from .monte_carlo import (
    _MCEuropeanValuation,
    _MCAmericanValuation,
    _MCAsianEuropeanValuation,
    _MCAsianAmericanValuation,
    _MCBarrierEuropeanValuation,
    _MCBarrierAmericanValuation,
)
from .binomial import (
    _BinomialEuropeanValuation,
    _BinomialAmericanValuation,
    _BinomialAsianValuation,
    _BinomialBarrierValuation,
)
from .bsm import _BSMEuropeanValuation
from .asian_analytical import _AnalyticalAsianValuation
from .barrier_analytical import _AnalyticalBarrierValuation, _is_triggered
from .pde import _FDEuropeanValuation, _FDAmericanValuation, _FDBarrierValuation
from ..rates import DiscountCurve
from ..market_environment import MarketData
from .contracts import AsianSpec, BarrierSpec, PayoffSpec, VanillaSpec
from .params import BinomialParams, MonteCarloParams, PDEParams, ValuationParams

logger = logging.getLogger(__name__)


# ── Implementation registries ───────────────────────────────────────
# Maps (PricingMethod, ExerciseType) → implementation class for vanilla specs.
_VANILLA_REGISTRY: dict[tuple[PricingMethod, ExerciseType], type] = {
    (PricingMethod.MONTE_CARLO, ExerciseType.EUROPEAN): _MCEuropeanValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.AMERICAN): _MCAmericanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.EUROPEAN): _BinomialEuropeanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.AMERICAN): _BinomialAmericanValuation,
    (PricingMethod.BSM, ExerciseType.EUROPEAN): _BSMEuropeanValuation,
    (PricingMethod.PDE_FD, ExerciseType.EUROPEAN): _FDEuropeanValuation,
    (PricingMethod.PDE_FD, ExerciseType.AMERICAN): _FDAmericanValuation,
}

# Maps (PricingMethod, ExerciseType) → implementation class for Asian option specs.
_ASIAN_REGISTRY: dict[tuple[PricingMethod, ExerciseType], type] = {
    (PricingMethod.MONTE_CARLO, ExerciseType.EUROPEAN): _MCAsianEuropeanValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.AMERICAN): _MCAsianAmericanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.EUROPEAN): _BinomialAsianValuation,
    (PricingMethod.BINOMIAL, ExerciseType.AMERICAN): _BinomialAsianValuation,
    (PricingMethod.BSM, ExerciseType.EUROPEAN): _AnalyticalAsianValuation,
}

# Maps (PricingMethod, ExerciseType) → implementation class for barrier option specs.
_BARRIER_REGISTRY: dict[tuple[PricingMethod, ExerciseType], type] = {
    (PricingMethod.BSM, ExerciseType.EUROPEAN): _AnalyticalBarrierValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.EUROPEAN): _MCBarrierEuropeanValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.AMERICAN): _MCBarrierAmericanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.EUROPEAN): _BinomialBarrierValuation,
    (PricingMethod.BINOMIAL, ExerciseType.AMERICAN): _BinomialBarrierValuation,
    (PricingMethod.PDE_FD, ExerciseType.EUROPEAN): _FDBarrierValuation,
    (PricingMethod.PDE_FD, ExerciseType.AMERICAN): _FDBarrierValuation,
}

# Maps GreekCalculationMethod → (required PricingMethod, capability_flag_name,
# human-readable str of supported greeks for that capability).
_GREEK_METHOD_RULES: dict[GreekCalculationMethod, tuple[PricingMethod, str, str]] = {
    GreekCalculationMethod.ANALYTICAL: (
        PricingMethod.BSM,
        "bsm_capable",  # always True for BSM — not a caller flag
        "all BSM greeks",
    ),
    GreekCalculationMethod.TREE: (
        PricingMethod.BINOMIAL,
        "tree_capable",
        "delta, gamma, and theta",
    ),
    GreekCalculationMethod.GRID: (
        PricingMethod.PDE_FD,
        "grid_capable",
        "delta, gamma, and theta",
    ),
}


@dataclass(frozen=True, slots=True)
class UnderlyingData:
    """Minimal underlying container for deterministic valuation methods.

    Parameterises the Black-Scholes-Merton (GBM) dynamics used by analytical,
    tree, and PDE engines.  When Monte Carlo simulation is required, use
    ``GBMProcess`` (or another ``PathSimulation`` subclass) instead — it
    carries the additional simulation configuration (paths, time grid, seed).

    Used by methods that do not require explicit path simulation (for example
    BSM, binomial trees, and PDE finite differences).

    Parameters
    ----------
    initial_value
        Spot value at pricing time.
    volatility
        Annualized volatility.
    market_data
        Market context containing pricing date, discount curve, and currency.
    discrete_dividends
        Optional sequence of ``(ex_date, amount)`` cash dividends.
    dividend_curve
        Optional dividend discount curve for modeling continuous yields.
    """

    initial_value: float
    volatility: float
    market_data: MarketData
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None
    dividend_curve: DiscountCurve | None = None

    def __post_init__(self) -> None:
        if self.discrete_dividends is not None:
            cleaned: list[tuple[dt.datetime, float]] = []
            for ex_date, amount in self.discrete_dividends:
                if not isinstance(ex_date, dt.datetime):
                    raise ConfigurationError(
                        "discrete_dividends entries must be (datetime, amount) tuples"
                    )
                try:
                    amt = float(amount)
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError("dividend amount must be numeric") from exc
                cleaned.append((ex_date, amt))
            object.__setattr__(
                self,
                "discrete_dividends",
                tuple(sorted(cleaned, key=lambda x: x[0])),
            )
        else:
            object.__setattr__(self, "discrete_dividends", ())

        if self.dividend_curve is not None and self.discrete_dividends:
            import warnings

            warnings.warn(
                "UnderlyingData: both dividend_curve and discrete_dividends "
                "provided. The continuous yield will enter the drift and discrete "
                "dividends will be subtracted at each ex-date.",
                stacklevel=2,
            )

    @property
    def pricing_date(self) -> dt.datetime:
        """Pricing date from ``market_data``."""
        return self.market_data.pricing_date

    @property
    def discount_curve(self) -> DiscountCurve:
        """Discount curve from ``market_data``."""
        return self.market_data.discount_curve

    @property
    def currency(self) -> str:
        """Currency from ``market_data``."""
        return self.market_data.currency

    @property
    def day_count_convention(self) -> DayCountConvention:
        """Day-count convention from ``market_data``."""
        return self.market_data.day_count_convention

    def replace(self, **kwargs: object) -> "UnderlyingData":
        """Create a new UnderlyingData instance with modified fields.

        This is used for bump-and-revalue calculations (e.g., Greeks) without
        mutating the original object, making it thread-safe and explicit.

        Parameters
        ----------
        **kwargs
            Fields to override (initial_value, volatility, dividend_curve,
            discrete_dividends, market_data)

        Returns
        -------
        UnderlyingData
            New instance with specified fields replaced
        """
        return dc_replace(self, **kwargs)  # type: ignore[arg-type]


def as_underlying_data(process: GBMProcess | UnderlyingData) -> UnderlyingData:
    """Convert a GBMProcess to an UnderlyingData instance.

    If *process* is already an ``UnderlyingData``, it is returned unchanged.
    This is useful when you have a Monte-Carlo process but need to run a
    deterministic method (BSM, binomial, PDE) that requires ``UnderlyingData``.

    Only ``GBMProcess`` is accepted — converting jump-diffusion or
    mean-reverting processes would silently discard non-GBM parameters.
    """
    if isinstance(process, UnderlyingData):
        return process
    if isinstance(process, GBMProcess):
        return UnderlyingData(
            initial_value=process.initial_value,
            volatility=process.volatility,
            market_data=process.market_data,
            dividend_curve=process.dividend_curve,
            discrete_dividends=process.discrete_dividends or None,
        )
    raise ConfigurationError(
        f"Expected UnderlyingData or GBMProcess, got {type(process).__name__}."
    )


def _memoize_result(fn):
    """Cache an :class:`OptionValuation` accessor's output on the instance.

    Keys on ``(fn.__name__, *sorted(kwargs.items()))``.  OptionValuation
    instances are effectively immutable — any "change" to inputs happens
    via :func:`dataclasses.replace` on the underlying/spec, yielding a
    fresh OV with its own empty cache — so output caching is safe and
    needs no explicit invalidation.  Exceptions propagate without being
    cached.  Internal calls like ``self.present_value()`` inside
    ``gamma``/``theta`` transparently benefit from a prior PV cache hit.

    Thread-safety: uses double-checked locking on a per-instance
    ``_cache_lock``.  The fast path (cache hit) is lock-free — it relies
    on the GIL to make dict lookup atomic — so concurrent cache hits
    don't serialise.  On a miss, a single thread computes under the lock
    while others wait and then read the cached result, preventing
    redundant solves when multiple threads query the same OV simultaneously.
    """

    @wraps(fn)
    def wrapper(self, **kwargs):
        key = (fn.__name__,) + tuple(sorted(kwargs.items()))
        cache = self._cache
        if key in cache:
            return cache[key]
        with self._cache_lock:
            if key in cache:
                return cache[key]
            result = fn(self, **kwargs)
            cache[key] = result
            return result

    return wrapper


class OptionValuation:
    """Single-factor option valuation facade and dispatcher.
    Instances are effectively immutable once created — constructor arguments are exposed as
    read-only properties."""

    def __init__(
        self,
        underlying: UnderlyingData | PathSimulation,
        spec: VanillaSpec | PayoffSpec | AsianSpec | BarrierSpec,
        pricing_method: PricingMethod,
        params: ValuationParams | None = None,
    ) -> None:
        # --- store private state ---
        self._spec = spec

        # Validate pricing_method early — comparisons rely on enum identity
        if not isinstance(pricing_method, PricingMethod):
            raise ConfigurationError(
                f"pricing_method must be PricingMethod enum, got {type(pricing_method).__name__}"
            )
        self._pricing_method = pricing_method

        # Resolve option_type (best-effort across spec variants)
        option_type = getattr(spec, "option_type", None)
        self._option_type: OptionType | None = (
            option_type if isinstance(option_type, OptionType) else None
        )

        # Resolve params
        self._params: ValuationParams | None = self._resolve_params(
            pricing_method=pricing_method, params=params, spec=spec
        )

        # --- currency resolution & check (default match) ---

        self._currency = spec.currency or underlying.currency

        if spec.currency is not None and spec.currency != underlying.currency:
            raise UnsupportedFeatureError(
                "Cross-currency valuation is not supported. "
                "Option currency must match the underlying market currency."
            )

        # Strategy guardrails
        if pricing_method is PricingMethod.BSM and self._option_type not in (
            OptionType.CALL,
            OptionType.PUT,
        ):
            raise UnsupportedFeatureError(
                f"{pricing_method.name} pricing requires a CALL or PUT option type "
                "(not available for custom PayoffSpec)."
            )

        # Optional sanity check: maturity must be after pricing date
        if self.maturity <= underlying.pricing_date:
            raise ValidationError("Option maturity must be after pricing_date.")

        # Validate that MC requires PathSimulation
        if pricing_method is PricingMethod.MONTE_CARLO and not isinstance(
            underlying, PathSimulation
        ):
            raise ConfigurationError(
                "Monte Carlo pricing requires underlying to be a PathSimulation instance"
            )

        # Validate that deterministic methods receive UnderlyingData
        if pricing_method in (
            PricingMethod.BINOMIAL,
            PricingMethod.BSM,
            PricingMethod.PDE_FD,
        ) and not isinstance(underlying, UnderlyingData):
            raise ConfigurationError(
                f"{pricing_method.name} pricing requires an UnderlyingData instance, "
                f"got {type(underlying).__name__}."
            )

        if (
            pricing_method is PricingMethod.BINOMIAL
            and isinstance(spec, BarrierSpec)
            and underlying.discrete_dividends
        ):
            raise UnsupportedFeatureError(
                "Binomial pricing of barrier options with discrete dividends is not supported. "
                "The escrowed-dividend tree adjustment used for vanilla options does not "
                "preserve the correct barrier-hit dynamics at ex-dividend dates. Accurate "
                "treatment would require a different tree construction rather than the "
                "standard recombining CRR barrier tree. Use PDE_FD or MONTE_CARLO instead."
            )

        # Assign early so helper methods (_asian_fixing_dates,
        # _barrier_monitoring_dates) can access self.pricing_date etc.
        # Overwritten below with the defensive copy for PathSimulation.
        self._underlying = underlying

        # Defensive copy: PathSimulation carries mutable simulation state
        # (time_grid, _last_normals) that is written during simulate().
        # Copying for thread-safety
        if isinstance(underlying, PathSimulation):
            sim_config = underlying._sim_config

            # Inject Asian observation dates into the copy's sim_config.
            if isinstance(spec, AsianSpec):
                fixing_dates = self._asian_fixing_dates()
                if fixing_dates[0] < underlying.pricing_date:
                    raise ValidationError(
                        "Asian fixing schedule must not start before pricing_date."
                    )
                extra = set(fixing_dates) - underlying.observation_dates
                if extra:
                    sim_config = dc_replace(
                        sim_config,
                        observation_dates=underlying.observation_dates | extra,
                    )

            # Inject barrier monitoring dates into the copy's sim_config.
            elif isinstance(spec, BarrierSpec):
                mon_dates = self._barrier_monitoring_dates()
                if mon_dates is not None:
                    if mon_dates[0] < underlying.pricing_date:
                        raise ValidationError(
                            "Barrier monitoring schedule must not start before pricing_date."
                        )
                    extra = set(mon_dates) - underlying.observation_dates
                    if extra:
                        sim_config = dc_replace(
                            sim_config,
                            observation_dates=underlying.observation_dates | extra,
                        )

            self._underlying = type(underlying)(
                market_data=underlying.market_data,
                process_params=underlying._process_params,
                sim_config=sim_config,
                corr=underlying.correlation_context,
                name=underlying.name,
            )

        elif isinstance(spec, AsianSpec):
            fixing_dates = self._asian_fixing_dates()
            if fixing_dates[0] < underlying.pricing_date:
                raise ValidationError("Asian fixing schedule must not start before pricing_date.")

        elif isinstance(spec, BarrierSpec):
            mon_dates = self._barrier_monitoring_dates()
            if mon_dates is not None and mon_dates[0] < underlying.pricing_date:
                raise ValidationError(
                    "Barrier monitoring schedule must not start before pricing_date."
                )

        # Dispatch to appropriate pricing method implementation
        self._impl = self._build_impl()

        # Output cache for repeated calls to PV / greek accessors.  Keyed
        # by (method_name, *sorted_kwargs).  See `_memoize_result` above.
        # The lock is per-instance (not class-level) so concurrent calls
        # on different OVs don't serialise — only concurrent calls on the
        # same OV coordinate to avoid redundant compute.
        self._cache: dict[tuple, Any] = {}
        # ``RLock`` (not ``Lock``) so the same thread can re-enter the
        # memoised accessors: e.g. ``gamma()`` internally calls
        # ``present_value()``, which would dead-lock a plain ``Lock``
        # because both hit ``_memoize_result`` and acquire this same lock.
        self._cache_lock = threading.RLock()

    # ──────────────────────────────
    # Public API (methods)
    # ──────────────────────────────

    @_memoize_result
    def present_value(self) -> float:
        """Calculate present value of the derivative."""
        base_pv = float(self._impl.present_value())
        if self._params is None or not getattr(self._params, "control_variate_european", False):
            return base_pv

        return float(self._apply_control_variate(base_pv))

    @_memoize_result
    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted pathwise present values (Monte Carlo only).

        The returned array is a read-only view over the engine's cached
        pathwise PVs.  The read-only flag is enforced by numpy
        (``ValueError`` on any attempted mutation) so the cache stays
        intact even if a caller attempts to modify the result.  Callers
        who need a mutable copy should call ``.copy()`` explicitly.
        """
        pv_pathwise = getattr(self._impl, "present_value_pathwise", None)
        if pv_pathwise is None:
            raise UnsupportedFeatureError(
                "present_value_pathwise is only implemented for Monte Carlo valuation."
            )
        arr = np.asarray(pv_pathwise())
        view = arr.view()
        view.flags.writeable = False
        return view

    @_memoize_result
    def delta(
        self,
        *,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        """Compute option delta.

        Parameters
        ----------
        epsilon
            Spot bump size used by central-difference numerical delta.
            Ignored for analytical, tree, pathwise, and likelihood-ratio methods.
            If ``None``, defaults to ``spot / 100``. For barrier options the
            bump is automatically shrunk if it would otherwise cross the
            barrier.
        greek_calc_method
            Greek computation method. When ``None``, the method is selected
            automatically from pricing-engine capabilities.

        Returns
        -------
        float
            First derivative of option value with respect to spot.
        """
        self._validate_bump(epsilon, greek_calc_method, "epsilon")
        method = self._resolve_greek_method(
            greek_calc_method,
            tree_capable=True,
            grid_capable=True,
        )
        if method is GreekCalculationMethod.PATHWISE:
            return float(self._impl.delta_pathwise())
        if method is GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.delta_lr())
        if method is not GreekCalculationMethod.NUMERICAL:
            return float(self._impl.delta())

        if epsilon is None:
            epsilon = self._underlying.initial_value / 100
        if isinstance(self._spec, BarrierSpec):
            epsilon = self._resolve_spot_bump(epsilon)

        s0 = self._underlying.initial_value
        up = self._bump_underlying(initial_value=s0 + epsilon)
        dn = self._bump_underlying(initial_value=s0 - epsilon)

        return (
            self._build_valuation(underlying=up).present_value()
            - self._build_valuation(underlying=dn).present_value()
        ) / (2 * epsilon)

    @_memoize_result
    def gamma(
        self,
        *,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        """Compute option gamma.

        Parameters
        ----------
        epsilon
            Spot bump size used for finite-difference estimation when required.
            If ``None``, defaults to ``spot / 100``.
        greek_calc_method
            Greek computation method. Supports analytical/tree methods where
            available, pathwise finite-difference for Monte Carlo, or numerical.

        Returns
        -------
        float
            Second derivative of option value with respect to spot.
        """
        self._validate_bump(
            epsilon,
            greek_calc_method,
            "epsilon",
            extra_allowed_methods=(GreekCalculationMethod.PATHWISE,),
        )
        method = self._resolve_greek_method(
            greek_calc_method,
            tree_capable=True,
            grid_capable=True,
        )
        if method is GreekCalculationMethod.PATHWISE:
            return float(self._impl.gamma_pathwise_fd(epsilon))
        if method is GreekCalculationMethod.LIKELIHOOD_RATIO:
            raise ValidationError(
                "likelihood_ratio is not available for gamma. "
                "Use PATHWISE (central-difference of pathwise delta) or NUMERICAL."
            )
        if method is not GreekCalculationMethod.NUMERICAL:
            return float(self._impl.gamma())

        if epsilon is None:
            epsilon = self._underlying.initial_value / 100
        if isinstance(self._spec, BarrierSpec):
            epsilon = self._resolve_spot_bump(epsilon)

        s0 = self._underlying.initial_value
        up = self._bump_underlying(initial_value=s0 + epsilon)
        dn = self._bump_underlying(initial_value=s0 - epsilon)

        value_right = self._build_valuation(underlying=up).present_value()
        value_left = self._build_valuation(underlying=dn).present_value()
        value_center = self.present_value()

        return (value_right - 2 * value_center + value_left) / (epsilon**2)

    @_memoize_result
    def vega(
        self,
        *,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        """Compute option vega.

        Parameters
        ----------
        epsilon
            Volatility bump used by central-difference numerical vega.
            If ``None``, defaults to ``0.01`` (a 1 vol-point bump).
        greek_calc_method
            Greek computation method. Supports analytical, pathwise, and
            likelihood-ratio methods where available; otherwise numerical.

        Returns
        -------
        float
            Vega reported per 1 vol-point (1%) change in volatility.
        """
        self._validate_bump(epsilon, greek_calc_method, "epsilon")
        method = self._resolve_greek_method(greek_calc_method)
        if method is GreekCalculationMethod.PATHWISE:
            return float(self._impl.vega_pathwise())
        if method is GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.vega_lr())
        if method is not GreekCalculationMethod.NUMERICAL:
            return float(self._impl.vega())

        if epsilon is None:
            epsilon = 0.01
        vol = self._underlying.volatility
        up = self._bump_underlying(volatility=vol + epsilon)
        dn = self._bump_underlying(volatility=vol - epsilon)

        vega = (
            (
                self._build_valuation(underlying=up).present_value()
                - self._build_valuation(underlying=dn).present_value()
            )
            / (2 * epsilon)
            / 100
        )
        return vega

    @_memoize_result
    def theta(
        self,
        *,
        time_bump_days: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        """Compute option theta.

        Parameters
        ----------
        greek_calc_method
            Greek computation method. Tree/analytical theta is used when available;
            otherwise bump-and-revalue is used.
        time_bump_days
            Calendar day bump applied to the pricing date for numerical theta.
            If ``None``, defaults to ``1.0``.

        Returns
        -------
        float
            Value change per day.
        """
        self._validate_bump(time_bump_days, greek_calc_method, "time_bump_days")

        # BSM analytical barrier theta: use the Black-Scholes PDE identity
        # (θ = rV − (r−q)Sδ − ½σ²S²γ) via the engine impl rather than a
        # forward-difference time bump (the former has better accuracy).
        # Only applies on auto-select; an explicit NUMERICAL request or a user-supplied
        # ``time_bump_days`` still routes to the original bump path.
        if (
            greek_calc_method is None
            and time_bump_days is None
            and self._pricing_method is PricingMethod.BSM
            and isinstance(self._spec, BarrierSpec)
        ):
            return float(self._impl.theta())

        method = self._resolve_greek_method(
            greek_calc_method,
            tree_capable=True,
            grid_capable=True,
        )
        if method is GreekCalculationMethod.PATHWISE:
            return float(self._impl.theta_pathwise())
        if method is GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.theta_lr())
        if method is not GreekCalculationMethod.NUMERICAL:
            return float(self._impl.theta())

        if time_bump_days is None:
            time_bump_days = 1.0
        bumped_date = self.pricing_date + dt.timedelta(days=time_bump_days)
        if bumped_date >= self.maturity:
            return 0.0

        value_now = self.present_value()

        underlying_bumped = self._bump_underlying(pricing_date=bumped_date)

        # Asian fixing-date awareness: if contractual observations fall before
        # the bumped pricing date they become observed ("seasoned") fixings.
        bumped_spec = self._spec
        if isinstance(self._spec, AsianSpec):
            bumped_spec = self._asian_theta_spec(bumped_date)

        value_bumped = self._build_valuation(
            underlying=underlying_bumped,
            spec=bumped_spec,
        ).present_value()

        return (value_bumped - value_now) / time_bump_days

    @_memoize_result
    def rho(
        self,
        *,
        rate_bump: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        """Compute option rho.

        Parameters
        ----------
        greek_calc_method
            Greek computation method. Analytical rho is used for BSM by default;
            otherwise finite-difference bump-and-revalue is used.
        rate_bump
            Absolute parallel bump in the continuously-compounded risk-free
            zero-rate curve for numerical rho. If ``None``, defaults to ``0.01``.

        Returns
        -------
        float
            Rho reported per 1% parallel rate move.
        """
        self._validate_bump(rate_bump, greek_calc_method, "rate_bump")
        # Rho is exempt from the barrier-binomial NUMERICAL guard: bumping
        # the rate does not change the Boyle-Lau barrier-aligned step count
        # (the formula depends only on σ, T and log(H/S)), so the up/down
        # trees share the same topology and the central difference is valid.
        method = self._resolve_greek_method(
            greek_calc_method, allow_barrier_binomial_numerical=True
        )
        if method is GreekCalculationMethod.PATHWISE:
            return float(self._impl.rho_pathwise())
        if method is GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.rho_lr())
        if method is not GreekCalculationMethod.NUMERICAL:
            return float(self._impl.rho())

        if rate_bump is None:
            rate_bump = 0.01
        curve_up = self.discount_curve.bump_parallel_zero_rate(rate_bump / 2)
        curve_down = self.discount_curve.bump_parallel_zero_rate(-rate_bump / 2)

        up = self._bump_underlying(discount_curve=curve_up)
        dn = self._bump_underlying(discount_curve=curve_down)

        return (
            (
                self._build_valuation(underlying=up).present_value()
                - self._build_valuation(underlying=dn).present_value()
            )
            / rate_bump
            * 0.01
        )

    # ──────────────────────────────
    # Read-only properties (public)
    # ──────────────────────────────

    @property
    def underlying(self) -> PathSimulation | UnderlyingData:
        """Underlying data/process used by this valuation instance."""
        return self._underlying

    @property
    def spec(self) -> VanillaSpec | PayoffSpec | AsianSpec | BarrierSpec:
        """Contract specification object for the valued instrument."""
        return self._spec

    @property
    def pricing_method(self) -> PricingMethod:
        """Pricing method selected for dispatch."""
        return self._pricing_method

    @property
    def params(self) -> ValuationParams | None:
        """Method-specific valuation parameters, if applicable."""
        return self._params

    @property
    def option_type(self) -> OptionType | None:
        """Resolved option type for vanilla-like specs, else ``None``."""
        return self._option_type

    @property
    def maturity(self) -> dt.datetime:
        """Contract maturity datetime."""
        return self._spec.maturity

    @property
    def strike(self) -> float | None:
        """Contract strike when defined on the specification."""
        return self._spec.strike

    @property
    def currency(self) -> str:
        """Valuation currency after constructor resolution/checks."""
        # effective currency resolved in __init__
        return self._currency

    @property
    def exercise_type(self) -> ExerciseType:
        """Exercise style of the contract."""
        return self._spec.exercise_type

    @property
    def contract_size(self) -> int | float:
        """Contract multiplier."""
        return self._spec.contract_size

    @property
    def pricing_date(self) -> dt.datetime:
        """Pricing date associated with underlying market data."""
        return self._underlying.pricing_date

    @property
    def discount_curve(self) -> DiscountCurve:
        """Discount curve used for valuation."""
        return self._underlying.discount_curve

    @property
    def day_count_convention(self) -> DayCountConvention:
        """Day-count basis used for date-to-year-fraction conversions."""
        return self._underlying.day_count_convention

    def __repr__(self) -> str:
        spec_type = type(self._spec).__name__
        strike = f", K={self.strike}" if self.strike is not None else ""
        option = f", {self._option_type.name}" if self._option_type is not None else ""
        return (
            f"OptionValuation({self._pricing_method.name}, "
            f"{self.exercise_type.name}{option}, "
            f"S={self._underlying.initial_value}{strike}, "
            f"T={self._fmt_dt(self.maturity)}, "
            f"pricing_date={self._fmt_dt(self.pricing_date)}, {spec_type})"
        )

    # ──────────────────────────────
    # Private API (helpers)
    # ──────────────────────────────
    def _maturity_year_fraction(self) -> float:
        """Time to maturity in years under the valuation day-count convention."""
        return float(
            calculate_year_fraction(
                self.pricing_date,
                self.maturity,
                day_count_convention=self.day_count_convention,
            )
        )

    @staticmethod
    def _fmt_dt(d: dt.datetime) -> str:
        """Format datetime as date-only when midnight, otherwise ISO."""
        if d.hour == 0 and d.minute == 0 and d.second == 0:
            return f"{d:%Y-%m-%d}"
        return d.isoformat()

    def _build_impl(self):
        spec = self._spec

        if isinstance(spec, BarrierSpec):
            impl_cls = _BARRIER_REGISTRY.get((self._pricing_method, spec.exercise_type))
            if impl_cls is None:
                raise UnsupportedFeatureError(
                    f"Barrier options with {spec.exercise_type.name} exercise "
                    f"do not support {self._pricing_method.name} pricing."
                )
            return impl_cls(self)

        if isinstance(spec, AsianSpec):
            impl_cls = _ASIAN_REGISTRY.get((self._pricing_method, spec.exercise_type))
            if impl_cls is None:
                raise ValidationError(
                    f"Asian options with {spec.exercise_type.name} exercise "
                    f"do not support {self._pricing_method.name} pricing."
                )
            return impl_cls(self)

        impl_cls = _VANILLA_REGISTRY.get((self._pricing_method, spec.exercise_type))
        if impl_cls is None:
            if self._pricing_method is PricingMethod.BSM:
                raise UnsupportedFeatureError(
                    "BSM is only applicable to European option valuation. "
                    "Select a different pricing method for American options such as Binomial, "
                    "PDE_FD or MONTE_CARLO."
                )
            raise ValidationError(
                f"{self._pricing_method.name} does not support {spec.exercise_type.name} exercise."
            )
        return impl_cls(self)

    @staticmethod
    def _resolve_params(
        *,
        pricing_method: PricingMethod,
        params: ValuationParams | None,
        spec: VanillaSpec | PayoffSpec | AsianSpec | BarrierSpec,
    ) -> ValuationParams | None:
        if params is None:
            if pricing_method is PricingMethod.MONTE_CARLO:
                return MonteCarloParams()
            if pricing_method is PricingMethod.BINOMIAL:
                if isinstance(spec, BarrierSpec):
                    # Continuous barriers get Boyle-Lau step inflation automatically,
                    # so 1000 base steps typically suffice.  Discrete barriers have
                    # no equivalent auto-adjustment (the tree is built exactly at
                    # the user-specified num_steps), so we default higher to
                    # accommodate the harder tree/monitoring-date alignment.
                    num_steps = 1000 if spec.monitoring is BarrierMonitoring.CONTINUOUS else 5000
                    return BinomialParams(num_steps=num_steps)
                return BinomialParams()
            if pricing_method is PricingMethod.PDE_FD:
                return PDEParams.for_barriers() if isinstance(spec, BarrierSpec) else PDEParams()
            return None

        if pricing_method is PricingMethod.MONTE_CARLO:
            if not isinstance(params, MonteCarloParams):
                raise ConfigurationError(
                    "pricing_method=MONTE_CARLO requires params=MonteCarloParams"
                )
            return params

        if pricing_method is PricingMethod.BINOMIAL:
            if not isinstance(params, BinomialParams):
                raise ConfigurationError("pricing_method=BINOMIAL requires params=BinomialParams")
            return params

        if pricing_method is PricingMethod.PDE_FD:
            if not isinstance(params, PDEParams):
                raise ConfigurationError("pricing_method=PDE_FD requires params=PDEParams")
            return params

        raise ConfigurationError(
            f"pricing_method={pricing_method.name} does not accept valuation params"
        )

    def _asian_fixing_dates(self) -> tuple[dt.datetime, ...]:
        """Resolve the contractual Asian fixing schedule as datetimes."""
        if not isinstance(self._spec, AsianSpec):
            raise ConfigurationError("Asian fixing schedule requested for non-Asian spec.")

        spec = self._spec
        if spec.fixing_dates is not None:
            return tuple(spec.fixing_dates)

        assert spec.num_observations is not None
        averaging_start = spec.averaging_start or self.pricing_date

        return tuple(
            pd.date_range(
                start=averaging_start,
                end=self.maturity,
                periods=spec.num_observations,
            ).to_pydatetime()
        )

    def _barrier_monitoring_dates(self) -> tuple[dt.datetime, ...] | None:
        """Resolve the barrier monitoring schedule as datetimes.

        Returns ``None`` for continuous monitoring (all grid points are used).
        """
        if not isinstance(self._spec, BarrierSpec):
            raise ConfigurationError("Barrier monitoring schedule requested for non-Barrier spec.")

        spec = self._spec
        if spec.monitoring is BarrierMonitoring.CONTINUOUS:
            return None

        if spec.monitoring_dates is not None:
            return tuple(spec.monitoring_dates)

        # N+1 dates from pricing_date to maturity; drop t=0 to leave N dates
        # at T/N, 2T/N, ..., T.  Matches the standard academic convention for
        # discretely-monitored barriers.
        return tuple(
            pd.date_range(
                start=self.pricing_date,
                end=self.maturity,
                periods=spec.num_observations + 1,
            )[1:].to_pydatetime()
        )

    def _barrier_triggered_at_inception(self) -> bool:
        """Return ``True`` only if the barrier has been hit AND that hit is
        observable at the pricing date.

        Continuous monitoring treats every instant as an observation, so a
        spot past the barrier at ``t=0`` is an immediate trigger.  Discrete
        monitoring only observes the barrier at explicit monitoring dates;
        the pricing date qualifies only if it appears in that schedule.
        """
        assert isinstance(self._spec, BarrierSpec), (
            "_barrier_triggered_at_inception called on non-BarrierSpec valuation; "
            "the dispatcher should route BarrierSpecs to barrier engines only."
        )
        spot = float(self._underlying.initial_value)
        if not _is_triggered(spot, self._spec.barrier, self._spec.direction):
            return False
        if self._spec.monitoring is BarrierMonitoring.CONTINUOUS:
            return True
        mon_dates = self._barrier_monitoring_dates()
        assert mon_dates is not None
        return any(d == self.pricing_date for d in mon_dates)

    def _apply_control_variate(self, base_pv: float) -> float:
        """Apply European control-variate adjustment to American base PV.

        Parameters
        ----------
        base_pv
            Raw American option present value from the selected numerical method.

        Returns
        -------
        float
            Control-variate adjusted present value.
        """
        if self._spec.exercise_type is not ExerciseType.AMERICAN:
            raise ValidationError(
                "control_variate_european is only valid for options with American exercise."
            )

        if isinstance(self._spec, AsianSpec):
            return self._apply_asian_control_variate(base_pv)

        if self._pricing_method not in (
            PricingMethod.BINOMIAL,
            PricingMethod.PDE_FD,
            PricingMethod.MONTE_CARLO,
        ):
            raise UnsupportedFeatureError(
                "control_variate_european is only supported for BINOMIAL, PDE_FD, "
                "and MONTE_CARLO pricing."
            )
        if isinstance(self._spec, PayoffSpec):
            raise UnsupportedFeatureError(
                "control_variate_european is not supported for PayoffSpec."
            )
        if self._option_type not in (OptionType.CALL, OptionType.PUT):
            raise UnsupportedFeatureError(
                "control_variate_european requires a CALL or PUT option type."
            )

        euro_spec = dc_replace(self._spec, exercise_type=ExerciseType.EUROPEAN)

        cv_params = dc_replace(self._params, control_variate_european=False)
        euro_num = OptionValuation(
            underlying=self._underlying,
            spec=euro_spec,
            pricing_method=self._pricing_method,
            params=cv_params,
        ).present_value()

        bsm_underlying = self._as_underlying_data()

        euro_bsm = OptionValuation(
            underlying=bsm_underlying,
            spec=euro_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()

        return base_pv + (euro_bsm - euro_num)

    def _as_underlying_data(self) -> UnderlyingData:
        """Return an UnderlyingData instance, extracting from GBMProcess if needed."""
        return as_underlying_data(self._underlying)  # type: ignore[arg-type]

    def _apply_asian_control_variate(self, base_pv: float) -> float:
        """Apply Asian-option European control-variate adjustment.

        Parameters
        ----------
        base_pv
            Raw American Asian present value from the selected numerical method.

        Returns
        -------
        float
            Control-variate adjusted present value.
        """
        if self._pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.MONTE_CARLO):
            raise UnsupportedFeatureError(
                "Asian control_variate_european is only supported for "
                "BINOMIAL and MONTE_CARLO pricing."
            )
        spec = self._spec
        assert isinstance(spec, AsianSpec)

        params = self._params

        if self._pricing_method is PricingMethod.BINOMIAL:
            assert isinstance(params, BinomialParams)
            if params.asian_tree_averages is None:
                raise UnsupportedFeatureError(
                    "Asian control_variate_european requires Hull tree averages "
                    "(set asian_tree_averages on BinomialParams)."
                )

        cv_params = dc_replace(params, control_variate_european=False)

        euro_spec = dc_replace(spec, exercise_type=ExerciseType.EUROPEAN)

        euro_num = OptionValuation(
            underlying=self._underlying,
            spec=euro_spec,
            pricing_method=self._pricing_method,
            params=cv_params,
        ).present_value()

        bsm_underlying = self._as_underlying_data()

        euro_analytical = OptionValuation(
            underlying=bsm_underlying,
            spec=euro_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()

        logger.debug(
            "Asian CV: american=%.6f euro_num=%.6f euro_analytical=%.6f adj=%.6f",
            base_pv,
            euro_num,
            euro_analytical,
            euro_analytical - euro_num,
        )

        return base_pv + (euro_analytical - euro_num)

    _MD_FIELDS = frozenset({"pricing_date", "discount_curve", "currency"})

    def _bump_underlying(
        self,
        **overrides: object,
    ) -> PathSimulation | UnderlyingData:
        """Return a new underlying with selected fields replaced.

        Callers pass attribute-level kwargs (``pricing_date=``,
        ``discount_curve=``, ``initial_value=``, ``volatility=``, etc.).
        Keys in ``_MD_FIELDS`` are routed into a bumped ``MarketData``;
        remaining keys are applied as direct-field overrides
        (``dc_replace`` for ``UnderlyingData``, ``process_params`` bump
        for ``PathSimulation``).
        """
        u = self._underlying

        # Split into MarketData-level vs direct-field overrides.
        md_kw: dict[str, Any] = {k: v for k, v in overrides.items() if k in self._MD_FIELDS}
        rest_kw: dict[str, Any] = {k: v for k, v in overrides.items() if k not in self._MD_FIELDS}

        if isinstance(u, UnderlyingData):
            if md_kw:
                rest_kw["market_data"] = dc_replace(u.market_data, **md_kw)
            return dc_replace(u, **rest_kw)

        # PathSimulation — rest_kw are process-param bumps.
        bumped_pp = dc_replace(u._process_params, **rest_kw) if rest_kw else u._process_params
        bumped_md = dc_replace(u.market_data, **md_kw) if md_kw else u.market_data

        return type(u)(  # type: ignore[arg-type]
            market_data=bumped_md,
            process_params=bumped_pp,
            sim_config=u._sim_config,
            corr=u.correlation_context,
            name=u.name,
        )

    @property
    def _is_mc_analytic_eligible(self) -> bool:
        """True when pathwise / likelihood-ratio MC Greeks are available."""
        return (
            self._pricing_method is PricingMethod.MONTE_CARLO
            and isinstance(self.underlying, GBMProcess)
            and isinstance(self._spec, VanillaSpec)
            and self._spec.exercise_type is ExerciseType.EUROPEAN
            and not self._underlying.discrete_dividends
        )

    def _resolve_greek_method(
        self,
        greek_calc_method: GreekCalculationMethod | None,
        *,
        tree_capable: bool = False,
        grid_capable: bool = False,
        allow_barrier_binomial_numerical: bool = False,
    ) -> GreekCalculationMethod:
        """Resolve and validate the Greek computation method.

        When *greek_calc_method* is ``None`` the best available method is
        chosen automatically (ANALYTICAL → TREE → GRID → PATHWISE → NUMERICAL).
        When an explicit method is supplied it is validated against the
        current pricing engine and capability flags.

        ``allow_barrier_binomial_numerical`` opts out of the final guard that
        blocks NUMERICAL bump-and-revalue greeks on the binomial engine for
        barrier options. Only rho is exempt — see the guard body for context.
        """
        if greek_calc_method is not None and not isinstance(
            greek_calc_method, GreekCalculationMethod
        ):
            raise ConfigurationError(
                f"greek_calc_method must be GreekCalculationMethod enum, "
                f"got {type(greek_calc_method).__name__}"
            )

        # --- auto-select when caller passes None ---
        if greek_calc_method is None:
            resolved = self._auto_select_greek_method(
                tree_capable=tree_capable,
                grid_capable=grid_capable,
            )
            self._reject_barrier_binomial_numerical(
                resolved, allow=allow_barrier_binomial_numerical
            )
            return resolved

        # --- validate explicit choice ---

        # Asian options only support NUMERICAL — no engine-native Greeks.
        if (
            isinstance(self._spec, AsianSpec)
            and greek_calc_method is not GreekCalculationMethod.NUMERICAL
        ):
            raise UnsupportedFeatureError(
                f"Asian options only support GreekCalculationMethod.NUMERICAL "
                f"(bump-and-revalue), got {greek_calc_method.name}."
            )

        # Barrier options have engine-native Greeks (TREE / GRID) and numerical
        # bump-and-revalue, but closed-form analytical Greeks are not currently
        # supported on the BSM analytical engine — _AnalyticalBarrierValuation
        # does not expose per-greek methods. Reject ANALYTICAL explicitly.
        if (
            isinstance(self._spec, BarrierSpec)
            and greek_calc_method is GreekCalculationMethod.ANALYTICAL
        ):
            raise UnsupportedFeatureError(
                "Barrier options do not support GreekCalculationMethod.ANALYTICAL. "
                "Use TREE (BINOMIAL), GRID (PDE_FD), or NUMERICAL (bump-and-revalue)."
            )

        capability_flags = {
            "tree_capable": tree_capable,
            "grid_capable": grid_capable,
            "bsm_capable": True,  # ANALYTICAL has no per-greek gate
        }

        rule = _GREEK_METHOD_RULES.get(greek_calc_method)
        if rule is not None:
            required_method, cap_flag, supported_greeks = rule
            if self._pricing_method is not required_method:
                raise ValidationError(
                    f"{greek_calc_method.value.capitalize()} greeks are only available for "
                    f"{required_method.name} pricing method."
                )
            if not capability_flags[cap_flag]:
                raise ValidationError(
                    f"{greek_calc_method.value.capitalize()} extraction is not available "
                    f"for this greek. Only {supported_greeks} support "
                    f"GreekCalculationMethod.{greek_calc_method.name}."
                )
        elif greek_calc_method in (
            GreekCalculationMethod.PATHWISE,
            GreekCalculationMethod.LIKELIHOOD_RATIO,
        ):
            self._validate_mc_greek_method(greek_calc_method)

        self._reject_barrier_binomial_numerical(
            greek_calc_method, allow=allow_barrier_binomial_numerical
        )
        return greek_calc_method

    def _reject_barrier_binomial_numerical(
        self,
        method: GreekCalculationMethod,
        *,
        allow: bool,
    ) -> None:
        """Block NUMERICAL bump-and-revalue greeks on binomial barrier specs.

        Bumping spot, volatility or time for a barrier option re-invokes
        ``_resolve_effective_num_steps`` on each bumped valuation, and the
        Boyle-Lau barrier-alignment formula
        ``candidate = i² σ² T / log(H/S)²`` depends on every one of those
        inputs.  The bumped trees therefore end up with *different* step
        counts from the center tree, so a central difference is comparing
        two unrelated tree topologies rather than approximating ``∂V/∂x``.
        Rho is exempt because the risk-free rate does not enter the
        Boyle-Lau formula, so rate bumps reuse the same tree
        topology and the finite difference is well-defined.
        """
        if allow:
            return
        if method is not GreekCalculationMethod.NUMERICAL:
            return
        if not isinstance(self._spec, BarrierSpec):
            return
        if self._pricing_method is not PricingMethod.BINOMIAL:
            return
        raise UnsupportedFeatureError(
            "Binomial NUMERICAL bump-and-revalue greeks are not supported "
            "for barrier options (rho is exempt). Bumping spot, volatility "
            "or time causes Boyle-Lau barrier alignment to pick a different "
            "tree step count for each bumped valuation, so the central "
            "difference compares two different tree topologies. "
            "Use GreekCalculationMethod.TREE for delta/gamma/theta, or "
            "switch to PricingMethod.PDE_FD for vega and for NUMERICAL "
            "bump-and-revalue on the full grid."
        )

    def _auto_select_greek_method(
        self,
        *,
        tree_capable: bool,
        grid_capable: bool,
    ) -> GreekCalculationMethod:
        """Choose the best Greek method for the current pricing engine."""
        # Asian options: no engine-native Greeks implemented — always bump-and-revalue.
        if isinstance(self._spec, AsianSpec):
            return GreekCalculationMethod.NUMERICAL
        if self._pricing_method is PricingMethod.BSM and not isinstance(self._spec, BarrierSpec):
            # Analytical Greeks not available for barriers
            return GreekCalculationMethod.ANALYTICAL
        if tree_capable and self._pricing_method is PricingMethod.BINOMIAL:
            return GreekCalculationMethod.TREE
        if grid_capable and self._pricing_method is PricingMethod.PDE_FD:
            return GreekCalculationMethod.GRID
        if self._is_mc_analytic_eligible:
            return GreekCalculationMethod.PATHWISE
        return GreekCalculationMethod.NUMERICAL

    def _validate_mc_greek_method(
        self,
        method: GreekCalculationMethod,
    ) -> None:
        """Validate PATHWISE / LIKELIHOOD_RATIO against MC prerequisites.

        Provides specific error messages for each failing condition.
        The fast-path check ``_is_mc_analytic_eligible`` covers the same
        conditions but returns a bool; this method gives actionable errors.
        """
        if self._pricing_method is not PricingMethod.MONTE_CARLO:
            raise ValidationError(
                f"{method.value} greeks are only available for MONTE_CARLO pricing method."
            )
        if not isinstance(self.underlying, GBMProcess):
            raise ValidationError("MC greeks are only available for GBMProcess underlying.")
        if not (
            isinstance(self._spec, VanillaSpec)
            and self._spec.exercise_type is ExerciseType.EUROPEAN
        ):
            raise ValidationError(
                f"{method.value} greeks are only implemented for "
                "vanilla European options (VanillaSpec)."
            )
        if self._underlying.discrete_dividends:
            raise UnsupportedFeatureError(
                "Pathwise and likelihood-ratio MC Greeks are not supported with discrete dividends."
            )

    # For barrier options, the spot bump epsilon is capped to
    # min(epsilon, _BARRIER_BUMP_MAX_FRACTION * |spot - barrier|)
    # so the bumped spot stays inside the alive region. 0.5 keeps the bump
    # at most halfway to the barrier.
    _BARRIER_BUMP_MAX_FRACTION: float = 0.5

    @staticmethod
    def _validate_bump(
        bump_value: float | None,
        greek_calc_method: GreekCalculationMethod | None,
        bump_name: str,
        extra_allowed_methods: tuple[GreekCalculationMethod, ...] = (),
    ) -> None:
        """Validate a numerical-greek bump argument.

        Two checks are bundled here so every greek entry point can run a
        single line:

        1. The bump must be strictly positive when supplied. Negative
           bumps are almost certainly user mistakes — central differences
           are sign-symmetric so a negative spot/vol/rate bump silently
           gives the same magnitude with confusing semantics, while a
           negative ``time_bump_days`` flips the forward-difference theta.
        2. The bump must be compatible with the chosen greek method.
           Passing ``epsilon=...`` together with an explicit non-NUMERICAL
           method is a contradiction — the bump would be silently ignored,
           letting users believe they are controlling it when they aren't.
           Auto-resolution (``greek_calc_method=None``) is exempt: users
           may always pass a default bump in case the resolved method ends
           up being NUMERICAL. ``extra_allowed_methods`` lets gamma also
           accept PATHWISE (which uses a finite-difference epsilon).
        """
        if bump_value is None:
            return
        if bump_value <= 0:
            raise ValidationError(f"{bump_name} must be strictly positive, got {bump_value}.")
        if greek_calc_method is None:
            return
        if greek_calc_method is GreekCalculationMethod.NUMERICAL:
            return
        if greek_calc_method in extra_allowed_methods:
            return
        raise ValidationError(
            f"{bump_name} is only used by NUMERICAL greeks; got "
            f"greek_calc_method={greek_calc_method.name}."
        )

    def _resolve_spot_bump(self, epsilon: float) -> float:
        """Cap ``epsilon`` so a central spot bump cannot cross the barrier.

        Caller must have verified ``self._spec`` is a :class:`BarrierSpec`.
        For barrier options, an unconstrained bump of ``s0 ± epsilon`` may
        cross the barrier on one side, putting the bumped option in a
        different (knocked-out vs alive) regime than the unbumped option.
        The resulting numerical greek then averages two regimes and is
        biased — sometimes by an order of magnitude.

        The returned bump is ``min(epsilon, max_fraction * |spot - barrier|)``
        where ``max_fraction = _BARRIER_BUMP_MAX_FRACTION``.
        """
        spec = self._spec
        assert isinstance(spec, BarrierSpec)
        s0 = float(self._underlying.initial_value)
        barrier = float(spec.barrier)
        if spec.direction is BarrierDirection.DOWN:
            gap = s0 - barrier
        else:
            gap = barrier - s0
        if gap <= 0:
            # Inception-triggered; let the engine handle it.
            return epsilon
        max_bump = gap * self._BARRIER_BUMP_MAX_FRACTION
        if epsilon <= max_bump:
            return epsilon
        logger.warning(
            "Numerical greek bump epsilon=%g would cross %s barrier H=%g "
            "(spot=%g); shrinking to %g.",
            epsilon,
            spec.direction.name,
            barrier,
            s0,
            max_bump,
        )
        return max_bump

    # ── Asian theta helpers ──────────────────────────────────────────────

    def _asian_theta_spec(self, bumped_date: dt.datetime) -> AsianSpec:
        """Build a seasoned AsianSpec for theta bump-and-revalue.

        When the pricing date is bumped forward, fixing dates that now fall
        on or before the new pricing date become observed fixings.

        Case 1 — fixing_date == original pricing_date:
            The observed price is deterministic (S₀).  A seasoned spec is
            returned with observed_average/observed_count set accordingly.
        Case 2 — pricing_date < fixing_date < bumped_date (intra-day):
            The fixing is stochastic from the current viewpoint and would
            require nested MC to handle correctly.  Raises
            ``UnsupportedFeatureError``.
        """
        spec = self._spec
        assert isinstance(spec, AsianSpec)

        # Build the effective contractual schedule from either explicit
        # fixing_dates or the num_observations schedule.
        schedule = self._asian_fixing_dates()
        elapsed = [d for d in schedule if d < bumped_date]
        if not elapsed:
            return spec

        # Case 2: intra-day fixings (between pricing_date and bumped_date,
        # but not equal to pricing_date) require nested MC.
        intraday = [d for d in elapsed if d != self.pricing_date]
        if intraday:
            raise UnsupportedFeatureError(
                f"Theta calculation is not supported when fixing dates fall "
                f"between pricing_date and the bumped date (intra-day fixings: "
                f"{[d.isoformat() for d in intraday]}). "
                f"This would require nested Monte Carlo simulation."
            )

        # Case 1: single elapsed fixing at pricing_date → deterministic S₀.
        assert len(elapsed) == 1
        s0 = float(self._underlying.initial_value)

        # Merge with any pre-existing seasoned state
        old_n1 = spec.observed_count or 0
        old_avg = spec.observed_average or 0.0
        new_n1 = old_n1 + 1
        if spec.averaging is AsianAveraging.GEOMETRIC and old_avg > 0.0:
            new_avg = float(np.exp((old_n1 * np.log(old_avg) + np.log(s0)) / new_n1))
        else:
            new_avg = (old_n1 * old_avg + s0) / new_n1

        future_fixings = tuple(d for d in schedule if d >= bumped_date)

        # Represent the seasoned remainder with explicit fixing_dates.
        # This works uniformly for both original schedule sources.
        return dc_replace(
            spec,
            num_observations=None,
            fixing_dates=future_fixings,
            observed_average=new_avg,
            observed_count=new_n1,
        )

    def _build_valuation(
        self,
        *,
        underlying,
        spec: VanillaSpec | PayoffSpec | AsianSpec | BarrierSpec | None = None,
    ) -> OptionValuation:
        return OptionValuation(
            underlying=underlying,
            spec=spec if spec is not None else self._spec,
            pricing_method=self._pricing_method,
            params=self._params,
        )
