"""Tests for Binomial tree option valuation."""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
)
from conftest import (
    BINOM_PARAMS,
    PRICING_DATE,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
    VOL,
)
from helpers import underlying, pv, spec
from derivatives_pricing.utils import calculate_year_fraction, expected_binomial_payoff
from derivatives_pricing.valuation import (
    BinomialParams,
    OptionValuation,
    VanillaSpec,
    UnderlyingData,
)
from derivatives_pricing.valuation.binomial import _BinomialBarrierValuation
from derivatives_pricing.valuation.contracts import BarrierSpec

# ---------------------------------------------------------------------------
# BSM reference (S=100, K=100, r=0.05, σ=0.20, T=1)
# ---------------------------------------------------------------------------
_BSM_ATM_CALL = 10.4506

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _binom(ud: UnderlyingData, sp: VanillaSpec, params: BinomialParams = BINOM_PARAMS) -> float:
    return pv(ud, sp, PricingMethod.BINOMIAL, params=params)


class TestBinomialValuation:
    """Tests for Binomial tree option valuation."""

    def test_binomial_european_call_atm(self):
        """Binomial European ATM call converges to BSM (500 steps)."""
        result = _binom(underlying(), spec())
        assert np.isclose(result, _BSM_ATM_CALL, rtol=0.005)

    def test_binomial_american_call_no_div_equal_to_european(self):
        """Test that American call >= European call (same parameters)."""
        eu_price = _binom(underlying(), spec())
        am_price = _binom(underlying(), spec(exercise=ExerciseType.AMERICAN))
        assert np.isclose(am_price, eu_price, rtol=0.005)

    def test_binomial_european_call_discrete_dividends_reduce_price(self):
        """Discrete dividends should reduce European call price in binomial tree."""
        sp = spec()
        pv_no_div = _binom(underlying(), sp)
        pv_div = _binom(
            underlying(discrete_dividends=[(PRICING_DATE + dt.timedelta(days=180), 1.0)]),
            sp,
        )
        assert pv_div < pv_no_div

    def test_binomial_american_put_early_exercise(self):
        """Test American put has early exercise premium."""
        eu_price = _binom(underlying(), spec(OptionType.PUT), params=BinomialParams(num_steps=100))
        am_price = _binom(
            underlying(),
            spec(OptionType.PUT, exercise=ExerciseType.AMERICAN),
            params=BinomialParams(num_steps=100),
        )
        assert am_price > eu_price

    def test_binomial_convergence(self):
        """More steps brings binomial price closer to BSM reference."""
        sp = spec()
        ud = underlying()
        price_100 = _binom(ud, sp, params=BinomialParams(num_steps=100))
        price_200 = _binom(ud, sp, params=BinomialParams(num_steps=200))
        # 200-step closer to BSM than 100-step
        assert abs(price_200 - _BSM_ATM_CALL) < abs(price_100 - _BSM_ATM_CALL)
        # both within 1% of BSM
        assert np.isclose(price_100, _BSM_ATM_CALL, rtol=0.01)
        assert np.isclose(price_200, _BSM_ATM_CALL, rtol=0.005)

    def test_binomial_pv_matches_expected_binomial_payoff(self):
        n_steps = 250
        pv_binom = _binom(underlying(), spec(), params=BinomialParams(num_steps=n_steps))

        T = calculate_year_fraction(
            PRICING_DATE, MATURITY, day_count_convention=DayCountConvention.ACT_365F
        )
        dt_step = T / n_steps
        u = np.exp(VOL * np.sqrt(dt_step))

        expected_payoff = expected_binomial_payoff(
            S0=SPOT,
            n=n_steps,
            T=T,
            option_type=OptionType.CALL,
            K=STRIKE,
            r=RATE,
            q=0,
            u=u,
        )
        pv_expected = np.exp(-RATE * T) * expected_payoff
        assert np.isclose(pv_binom, pv_expected, rtol=1.0e-4)

    @pytest.mark.parametrize(
        ("monitoring", "num_observations", "expected_adjusted_steps"),
        [
            (BarrierMonitoring.CONTINUOUS, None, True),
            (BarrierMonitoring.DISCRETE, 12, False),
        ],
    )
    def test_binomial_barrier_uses_correct_effective_steps(
        self,
        monitoring: BarrierMonitoring,
        num_observations: int | None,
        expected_adjusted_steps: bool,
    ):
        barrier_spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
            barrier=95.0,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            monitoring=monitoring,
            num_observations=num_observations,
        )
        valuation = OptionValuation(
            underlying(),
            barrier_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=25),
        )
        engine = _BinomialBarrierValuation(valuation)

        effective_steps = engine._effective_num_steps()
        if expected_adjusted_steps:
            assert effective_steps > 25
        else:
            assert effective_steps == 25

        discount_factors, _, _ = engine._setup_binomial_parameters()
        assert discount_factors.shape == (effective_steps,)

        solved_lattice = engine.solve()

        assert solved_lattice.shape == (effective_steps + 1, effective_steps + 1)

    def test_binomial_knock_in_tree_greeks_use_inactive_lattice(self):
        barrier_spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
            barrier=115.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        valuation = OptionValuation(
            underlying(),
            barrier_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=200),
        )
        engine = _BinomialBarrierValuation(valuation)

        _, inactive = engine._solve_knock_in(early_exercise=False)
        _, _, spot_lattice = engine._setup_binomial_parameters()
        T = valuation._maturity_year_fraction()
        dt = T / engine._effective_num_steps()

        expected_delta = (inactive[0, 1] - inactive[1, 1]) / (
            spot_lattice[0, 1] - spot_lattice[1, 1]
        )
        delta_up = (inactive[0, 2] - inactive[1, 2]) / (spot_lattice[0, 2] - spot_lattice[1, 2])
        delta_down = (inactive[1, 2] - inactive[2, 2]) / (spot_lattice[1, 2] - spot_lattice[2, 2])
        h = (spot_lattice[0, 2] - spot_lattice[2, 2]) / 2.0
        expected_gamma = (delta_up - delta_down) / h
        expected_theta = ((inactive[1, 2] - inactive[0, 0]) / (2.0 * dt)) / 365.0

        assert np.isclose(
            valuation.delta(greek_calc_method=GreekCalculationMethod.TREE),
            expected_delta,
        )
        assert np.isclose(
            valuation.gamma(greek_calc_method=GreekCalculationMethod.TREE),
            expected_gamma,
        )
        assert np.isclose(
            valuation.theta(greek_calc_method=GreekCalculationMethod.TREE),
            expected_theta,
        )

    def test_binomial_knock_in_tree_greeks_use_vanilla_lattice_when_triggered(self):
        barrier_spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
            barrier=95.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        valuation = OptionValuation(
            underlying(),
            barrier_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=200),
        )
        engine = _BinomialBarrierValuation(valuation)

        vanilla_lattice = engine._solve_backward(early_exercise=False)
        _, _, spot_lattice = engine._setup_binomial_parameters()
        T = valuation._maturity_year_fraction()
        dt = T / engine._effective_num_steps()

        expected_delta = (vanilla_lattice[0, 1] - vanilla_lattice[1, 1]) / (
            spot_lattice[0, 1] - spot_lattice[1, 1]
        )
        delta_up = (vanilla_lattice[0, 2] - vanilla_lattice[1, 2]) / (
            spot_lattice[0, 2] - spot_lattice[1, 2]
        )
        delta_down = (vanilla_lattice[1, 2] - vanilla_lattice[2, 2]) / (
            spot_lattice[1, 2] - spot_lattice[2, 2]
        )
        h = (spot_lattice[0, 2] - spot_lattice[2, 2]) / 2.0
        expected_gamma = (delta_up - delta_down) / h
        expected_theta = ((vanilla_lattice[1, 2] - vanilla_lattice[0, 0]) / (2.0 * dt)) / 365.0

        assert np.isclose(
            valuation.delta(greek_calc_method=GreekCalculationMethod.TREE),
            expected_delta,
        )
        assert np.isclose(
            valuation.gamma(greek_calc_method=GreekCalculationMethod.TREE),
            expected_gamma,
        )
        assert np.isclose(
            valuation.theta(greek_calc_method=GreekCalculationMethod.TREE),
            expected_theta,
        )
