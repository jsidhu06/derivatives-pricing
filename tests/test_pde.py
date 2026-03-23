"""PDE FD grid/method/solver equivalence tests.

Verifies that different PDE finite-difference schemes (Explicit, Implicit,
Crank-Nicolson, Explicit-Hull) and space grids (SPOT, LOG_SPOT) produce
consistent prices.  These are internal engine tests — cross-method and
QuantLib comparisons live in test_quantlib_comparison.py.
"""

import numpy as np
import pytest

from derivatives_pricing.exceptions import StabilityError, UnsupportedFeatureError
from derivatives_pricing.enums import (
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
)
from derivatives_pricing.valuation.contracts import PayoffSpec
from helpers import (
    flat_curve,
    underlying,
    spec,
    PRICING_DATE,
    MATURITY,
)
from derivatives_pricing.valuation import OptionValuation
from derivatives_pricing.valuation.params import PDEParams


def test_pde_fd_grid_method_equivalence_european():
    """PDE FD variants should be in the same neighborhood for European options."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
    ud = underlying(initial_value=100.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=160,
                time_steps=240,
                method=method,
                space_grid=grid,
                american_solver=PDEEarlyExercise.INTRINSIC,
            )

            if (
                method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                and grid is PDESpaceGrid.SPOT
            ):
                with pytest.raises(
                    StabilityError, match="Explicit spot-grid scheme likely unstable"
                ):
                    OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                continue

            pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
            assert np.isclose(pv, baseline, rtol=0.005)


def test_pde_fd_grid_method_equivalence_american():
    """PDE FD American variants should be in the same neighborhood."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
    ud = underlying(initial_value=95.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            for solver in (PDEEarlyExercise.INTRINSIC, PDEEarlyExercise.GAUSS_SEIDEL):
                params = PDEParams(
                    spot_steps=160,
                    time_steps=240,
                    method=method,
                    space_grid=grid,
                    american_solver=solver,
                    max_iter=20_000,
                )

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and solver is PDEEarlyExercise.GAUSS_SEIDEL
                ):
                    with pytest.raises(
                        UnsupportedFeatureError, match="GAUSS_SEIDEL is not supported"
                    ):
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)


class TestPDEGridTheta:
    """Verify PDE grid theta sign and magnitude against BSM analytical theta."""

    @pytest.fixture(autouse=True)
    def setup(self):
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
        self.ud = underlying(initial_value=100.0, volatility=0.20, dividend_curve=q_curve)
        self.pde_params = PDEParams(spot_steps=400, time_steps=400)

    def test_call_grid_theta_negative(self):
        spec_call = spec(strike=100.0, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)
        ov = OptionValuation(self.ud, spec_call, PricingMethod.PDE_FD, params=self.pde_params)
        theta = ov.theta(greek_calc_method=GreekCalculationMethod.GRID)
        assert theta < 0

    def test_put_grid_theta_negative(self):
        spec_put = spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.EUROPEAN)
        ov = OptionValuation(self.ud, spec_put, PricingMethod.PDE_FD, params=self.pde_params)
        theta = ov.theta(greek_calc_method=GreekCalculationMethod.GRID)
        assert theta < 0

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_grid_theta_close_to_bsm(self, option_type: OptionType):
        spec_vanilla = spec(strike=100.0, option_type=option_type, exercise=ExerciseType.EUROPEAN)
        bsm = OptionValuation(self.ud, spec_vanilla, PricingMethod.BSM)
        pde = OptionValuation(self.ud, spec_vanilla, PricingMethod.PDE_FD, params=self.pde_params)

        theta_bsm = bsm.theta()
        theta_pde = pde.theta(greek_calc_method=GreekCalculationMethod.GRID)

        assert np.isclose(theta_pde, theta_bsm, rtol=0.02)


##### CUSTOM PAYOFF TESTS #####


def _bull_call_spread(S):
    return np.maximum(S - 95.0, 0) - np.maximum(S - 115.0, 0)


def _capped_strangle(S):
    return np.minimum(40.0, np.maximum(90.0 - S, 0) + np.maximum(S - 110.0, 0))


_CUSTOM_PAYOFFS = [
    pytest.param(_bull_call_spread, id="bull_call_spread"),
    pytest.param(_capped_strangle, id="capped_strangle"),
]


class TestPDECustomPayoffMethodEquivalence:
    """PDE FD variants should agree for custom PayoffSpec payoffs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
        self.ud = underlying(initial_value=100.0, dividend_curve=q_curve)

    @pytest.mark.parametrize("payoff_fn", _CUSTOM_PAYOFFS)
    def test_european(self, payoff_fn):
        sp = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=MATURITY,
            payoff_fn=payoff_fn,
        )
        base_params = PDEParams(spot_steps=400, time_steps=400)
        baseline = OptionValuation(
            self.ud, sp, PricingMethod.PDE_FD, params=base_params
        ).present_value()

        for method in (
            PDEMethod.IMPLICIT,
            PDEMethod.EXPLICIT,
            PDEMethod.EXPLICIT_HULL,
            PDEMethod.CRANK_NICOLSON,
        ):
            for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
                params = PDEParams(
                    spot_steps=400,
                    time_steps=400,
                    method=method,
                    space_grid=grid,
                )

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(
                            self.ud, sp, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(
                    self.ud, sp, PricingMethod.PDE_FD, params=params
                ).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)

    @pytest.mark.parametrize("payoff_fn", _CUSTOM_PAYOFFS)
    def test_american(self, payoff_fn):
        sp = PayoffSpec(
            exercise_type=ExerciseType.AMERICAN,
            maturity=MATURITY,
            payoff_fn=payoff_fn,
        )
        base_params = PDEParams(spot_steps=400, time_steps=400)
        baseline = OptionValuation(
            self.ud, sp, PricingMethod.PDE_FD, params=base_params
        ).present_value()

        for method in (
            PDEMethod.IMPLICIT,
            PDEMethod.EXPLICIT,
            PDEMethod.EXPLICIT_HULL,
            PDEMethod.CRANK_NICOLSON,
        ):
            for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
                is_explicit = method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                params = PDEParams(
                    spot_steps=400,
                    time_steps=400,
                    method=method,
                    space_grid=grid,
                    american_solver=(
                        PDEEarlyExercise.INTRINSIC if is_explicit else PDEEarlyExercise.GAUSS_SEIDEL
                    ),
                )

                if is_explicit and grid is PDESpaceGrid.SPOT:
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(
                            self.ud, sp, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(
                    self.ud, sp, PricingMethod.PDE_FD, params=params
                ).present_value()
                assert np.isclose(pv, baseline, rtol=0.02)
