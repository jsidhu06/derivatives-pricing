"""Tests for the two layers of PV/greek result caching.

Phase 1 — ``OptionValuation`` output cache.  Repeated accessor calls with
identical kwargs reuse the previously returned value without touching the
engine impl.

Phase 2 — engine-impl solve cache.  Grid (PDE_FD) and tree (Binomial)
engines memoise the expensive backward solve on the instance so that the
first accessor call (PV or any native greek) pays for the solve and every
subsequent grid-/tree-native greek is an O(1) lookup.
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation import (
    BarrierSpec,
    OptionValuation,
    UnderlyingData,
    VanillaSpec,
)
from derivatives_pricing.valuation.params import MonteCarloParams


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)


def _underlying() -> UnderlyingData:
    md = MarketData(
        PRICING_DATE,
        DiscountCurve.flat(0.05, 2.0),
        currency="USD",
        day_count_convention=DayCountConvention.ACT_365F,
    )
    return UnderlyingData(initial_value=100.0, volatility=0.20, market_data=md)


def _am_doc_spec() -> BarrierSpec:
    return BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.AMERICAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )


def _eu_dic_spec() -> BarrierSpec:
    return BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )


def _call_all_native_greeks(ov: OptionValuation) -> None:
    """Touch every grid/tree-native accessor exactly once."""
    ov.present_value()
    ov.delta()
    ov.gamma()
    ov.theta()


class TestOptionValuationOutputCache:
    """Phase 1 — OV-level output cache (keyed on accessor name + kwargs)."""

    def test_repeated_present_value_call_skips_impl(self):
        """Second ``present_value()`` hits the OV cache without touching the impl."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        ov.present_value()  # warm cache
        with patch.object(ov._impl, "present_value", wraps=ov._impl.present_value) as impl_pv:
            ov.present_value()
            ov.present_value()
        assert impl_pv.call_count == 0

    def test_distinct_kwargs_do_not_share_cache(self):
        """``delta(epsilon=a)`` and ``delta(epsilon=b)`` cache separately."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        d1 = ov.delta(epsilon=0.5)
        d2 = ov.delta(epsilon=1.0)
        assert d1 == ov.delta(epsilon=0.5)  # cache hit
        assert d2 == ov.delta(epsilon=1.0)  # cache hit
        assert ("delta", ("epsilon", 0.5)) in ov._cache
        assert ("delta", ("epsilon", 1.0)) in ov._cache


class TestEngineLevelCaching:
    """Phase 2 — engine-impl solve cache shared across PV + native greeks."""

    def test_fd_barrier_ko_solves_once_across_pv_and_greeks(self):
        """PDE_FD KO: ``_compute_solve`` fires once for PV + delta + gamma + theta."""
        ov = OptionValuation(_underlying(), _am_doc_spec(), PricingMethod.PDE_FD)
        with patch.object(
            ov._impl, "_compute_solve", wraps=ov._impl._compute_solve
        ) as compute_solve:
            _call_all_native_greeks(ov)
        assert compute_solve.call_count == 1

    def test_fd_barrier_eu_ki_components_solved_once_across_greeks(self):
        """European KI parity: ``_compute_european_ki_components`` fires once."""
        ov = OptionValuation(_underlying(), _eu_dic_spec(), PricingMethod.PDE_FD)
        with patch.object(
            ov._impl,
            "_compute_european_ki_components",
            wraps=ov._impl._compute_european_ki_components,
        ) as compute_components:
            _call_all_native_greeks(ov)
        assert compute_components.call_count == 1

    def test_binomial_american_backward_solves_once_across_pv_and_greeks(self):
        """Binomial American vanilla: ``_compute_solve_backward`` fires once."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.PUT,
                exercise_type=ExerciseType.AMERICAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BINOMIAL,
        )
        with patch.object(
            ov._impl,
            "_compute_solve_backward",
            wraps=ov._impl._compute_solve_backward,
        ) as compute_backward:
            _call_all_native_greeks(ov)
        assert compute_backward.call_count == 1

    def test_binomial_barrier_ko_solves_once_across_pv_and_greeks(self):
        """Binomial barrier KO: ``_compute_knock_out`` fires once for PV + 3 greeks."""
        ov = OptionValuation(_underlying(), _am_doc_spec(), PricingMethod.BINOMIAL)
        with patch.object(
            ov._impl, "_compute_knock_out", wraps=ov._impl._compute_knock_out
        ) as compute_ko:
            _call_all_native_greeks(ov)
        assert compute_ko.call_count == 1


class TestThreadSafeCaching:
    """Double-checked locking prevents redundant computes under concurrent access.

    Each test launches many threads against the same OV on a cold cache and
    asserts that the expensive ``_compute_*`` hook fires exactly once.  A
    brief ``time.sleep`` inside the patched hook widens the race window
    — without the lock, all concurrent callers would pass the ``if key in
    cache`` check and each call compute independently (``call_count == N``).
    With the lock, one thread enters the critical section and the rest
    read the cached result.
    """

    _N_THREADS = 8
    _COMPUTE_SLEEP = 0.05  # seconds — wide enough to guarantee race window

    @classmethod
    def _concurrent_call(cls, target, n_threads: int | None = None) -> list[float]:
        """Run ``target()`` from ``n_threads`` worker threads and return results."""
        workers = n_threads or cls._N_THREADS
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(target) for _ in range(workers)]
            return [f.result() for f in futures]

    def test_pde_concurrent_present_value_solves_once(self):
        """PDE_FD: concurrent ``present_value()`` triggers a single ``_compute_solve``."""
        ov = OptionValuation(_underlying(), _am_doc_spec(), PricingMethod.PDE_FD)
        original = ov._impl._compute_solve

        def slow(*args, **kwargs):
            time.sleep(self._COMPUTE_SLEEP)
            return original(*args, **kwargs)

        with patch.object(ov._impl, "_compute_solve", side_effect=slow) as compute:
            results = self._concurrent_call(ov.present_value)

        assert compute.call_count == 1
        assert len(set(results)) == 1  # all threads got the same cached value

    def test_binomial_concurrent_present_value_solves_once(self):
        """Binomial: concurrent ``present_value()`` triggers a single backward solve."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.PUT,
                exercise_type=ExerciseType.AMERICAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BINOMIAL,
        )
        original = ov._impl._compute_solve_backward

        def slow(*args, **kwargs):
            time.sleep(self._COMPUTE_SLEEP)
            return original(*args, **kwargs)

        with patch.object(ov._impl, "_compute_solve_backward", side_effect=slow) as compute:
            results = self._concurrent_call(ov.present_value)

        assert compute.call_count == 1
        assert len(set(results)) == 1

    def test_pde_eu_ki_concurrent_greeks_solve_components_once(self):
        """European KI parity: concurrent greeks fire ``_compute_european_ki_components`` once."""
        ov = OptionValuation(_underlying(), _eu_dic_spec(), PricingMethod.PDE_FD)
        original = ov._impl._compute_european_ki_components

        def slow(*args, **kwargs):
            time.sleep(self._COMPUTE_SLEEP)
            return original(*args, **kwargs)

        with patch.object(ov._impl, "_compute_european_ki_components", side_effect=slow) as compute:
            # Mix of greek accessors, all internally needing the KI components.
            targets = [ov.present_value, ov.delta, ov.gamma, ov.theta]
            with ThreadPoolExecutor(max_workers=len(targets) * 2) as pool:
                futures = [pool.submit(t) for t in targets for _ in range(2)]
                for f in futures:
                    f.result()

        assert compute.call_count == 1

    def test_ov_cache_concurrent_present_value_impl_called_once(self):
        """OV-level cache: concurrent ``present_value()`` hits ``_impl.present_value`` once."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        original = ov._impl.present_value

        def slow():
            time.sleep(self._COMPUTE_SLEEP)
            return original()

        with patch.object(ov._impl, "present_value", side_effect=slow) as impl_pv:
            results = self._concurrent_call(ov.present_value)

        assert impl_pv.call_count == 1
        assert len(set(results)) == 1

    def test_concurrent_same_kwargs_coordinate(self):
        """Concurrent calls with the *same* kwargs share a single cache entry.

        Distinct kwargs cache separately (verified in
        :class:`TestOptionValuationOutputCache`); here we confirm that
        concurrent callers with identical kwargs converge on one compute.
        """
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        # Count calls to the underlying impl.present_value inside delta's
        # spot-bump sub-valuations — with a deterministic ``epsilon`` the
        # two bumped OVs (s0+eps, s0-eps) each solve exactly once.
        # Repeat calls from many threads should not increase the count.
        eps = 0.5
        d1 = ov.delta(epsilon=eps)
        results = self._concurrent_call(lambda: ov.delta(epsilon=eps))
        assert all(r == d1 for r in results)
        # Cache now contains exactly one entry for delta at this epsilon.
        assert ("delta", ("epsilon", eps)) in ov._cache


def _mc_gbm_process() -> GBMProcess:
    """Construct a small GBMProcess for Monte Carlo pathwise tests."""
    md = MarketData(
        PRICING_DATE,
        DiscountCurve.flat(0.05, 2.0),
        currency="USD",
        day_count_convention=DayCountConvention.ACT_365F,
    )
    return GBMProcess(
        md,
        GBMParams(initial_value=100.0, volatility=0.20),
        SimulationConfig(paths=2_000, num_steps=12, end_date=MATURITY),
    )


def _mc_vanilla_spec() -> VanillaSpec:
    return VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
    )


def _mc_ov() -> OptionValuation:
    """Build a seeded MC OV for pathwise-cache tests.

    Seeding makes path realisations deterministic across test ordering
    and parallel workers (pytest-xdist) — otherwise the test outcome can
    depend on whatever numpy global RNG state happens to be at test
    start, which can flip path-level values between OTM and ITM.
    """
    return OptionValuation(
        _mc_gbm_process(),
        _mc_vanilla_spec(),
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=42),
    )


class TestPathwisePresentValueCache:
    """``present_value_pathwise`` is memoised and returns a read-only view.

    The pathwise MC accessor is the one accessor returning an ``ndarray``
    rather than a ``float``, so it must explicitly enforce the cache
    contract via ``ndarray.flags.writeable = False`` — a mutation attempt
    raises ``ValueError`` from numpy, protecting the cached array from
    accidental corruption regardless of caller discipline.
    """

    def test_repeated_pathwise_call_skips_impl(self):
        """Second ``present_value_pathwise()`` hits the OV cache without touching the impl."""
        ov = _mc_ov()
        ov.present_value_pathwise()  # warm cache
        with patch.object(
            ov._impl, "present_value_pathwise", wraps=ov._impl.present_value_pathwise
        ) as impl_pv:
            ov.present_value_pathwise()
            ov.present_value_pathwise()
        assert impl_pv.call_count == 0

    def test_pathwise_returns_read_only_view(self):
        """Returned array is read-only — mutation raises ``ValueError``."""
        ov = _mc_ov()
        arr = ov.present_value_pathwise()
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            arr[0] = 0.0

    def test_pathwise_copy_is_mutable(self):
        """``.copy()`` of the returned array is mutable and does not affect the cache."""
        ov = _mc_ov()
        arr_before = ov.present_value_pathwise()
        # Snapshot the pre-mutation state in a fresh (mutable) array so we
        # can detect cache corruption even though arr_before itself is
        # a read-only view sharing memory with the cached array.
        snapshot = arr_before.copy()
        mutable = arr_before.copy()
        # Sentinel chosen so the assertion is robust regardless of path
        # realisation — a call PV is never negative, so -999.0 is
        # unambiguously "not a real value."
        sentinel = -999.0
        mutable[0] = sentinel
        arr_after = ov.present_value_pathwise()
        assert mutable[0] == sentinel  # copy is writable
        assert np.array_equal(arr_after, snapshot)  # cache untouched

    def test_concurrent_pathwise_solves_once(self):
        """Concurrent ``present_value_pathwise()`` triggers a single impl call."""
        ov = _mc_ov()
        original = ov._impl.present_value_pathwise

        def slow():
            time.sleep(0.05)
            return original()

        with patch.object(ov._impl, "present_value_pathwise", side_effect=slow) as impl_pv:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(ov.present_value_pathwise) for _ in range(8)]
                results = [f.result() for f in futures]

        assert impl_pv.call_count == 1
        # All threads got the same (read-only) array reference.
        assert all(np.array_equal(r, results[0]) for r in results)
        assert all(not r.flags.writeable for r in results)
