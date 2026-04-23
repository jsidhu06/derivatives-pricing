"""Comprehensive tests for greek calculations (delta, gamma, vega, theta, rho)."""

import datetime as dt
import logging
import numpy as np
import pytest

from derivatives_pricing.exceptions import (
    ConfigurationError,
    ValidationError,
    UnsupportedFeatureError,
)
from derivatives_pricing.enums import (
    AsianAveraging,
    BarrierDirection,
    BarrierAction,
    BarrierMonitoring,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import assert_greeks_close, flat_curve
from derivatives_pricing.utils import calculate_year_fraction
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    PathSimulation,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation import (
    AsianSpec,
    BarrierSpec,
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)
from derivatives_pricing.valuation import BinomialParams, MonteCarloParams, PDEParams

logger = logging.getLogger(__name__)


class TestGreeksSetup:
    """Base setup for greek tests with common parameters + factory helpers.

    Notes
    -----
    - setup fixture is function-scoped (default), so each test method gets fresh state.
    - factories avoid mutating shared UnderlyingData instances within a test.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.spot = 100.0
        self.strike = 100.0
        self.rate = 0.05
        self.volatility = 0.20
        self.currency = "USD"

        # Discount curve
        self.csr = flat_curve(self.pricing_date, self.maturity, self.rate)

        # Market data + sim_config config for Monte Carlo
        self.market_data = MarketData(self.pricing_date, self.csr, currency=self.currency)
        self.sim_config = SimulationConfig(
            paths=200_000,
            num_steps=52,
            end_date=self.maturity,
        )

    # -------- factories (preferred over mutating shared instances) --------

    def _make_ud(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        dividend_curve: DiscountCurve | None = None,
        pricing_date: dt.datetime | None = None,
    ) -> UnderlyingData:
        md = (
            self.market_data
            if pricing_date is None
            else MarketData(pricing_date, self.csr, currency=self.currency)
        )
        return UnderlyingData(
            initial_value=self.spot if spot is None else spot,
            volatility=self.volatility if vol is None else vol,
            market_data=md,
            dividend_curve=dividend_curve,
        )

    def _make_q_curve(
        self,
        dividend_rate: float,
        pricing_date: dt.datetime | None = None,
    ) -> DiscountCurve | None:
        if dividend_rate == 0.0:
            return None
        base_date = self.pricing_date if pricing_date is None else pricing_date
        return flat_curve(base_date, self.maturity, dividend_rate)

    def _make_spec(
        self,
        *,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        strike: float | None = None,
        maturity: dt.datetime | None = None,
        currency: str | None = None,
    ) -> VanillaSpec:
        return VanillaSpec(
            option_type=option_type,
            exercise_type=exercise_type,
            strike=self.strike if strike is None else strike,
            maturity=self.maturity if maturity is None else maturity,
            currency=self.currency if currency is None else currency,
        )

    def _make_val(
        self,
        underlying: PathSimulation | UnderlyingData,
        spec: VanillaSpec,
        method: PricingMethod,
        params=None,
    ) -> OptionValuation:
        return OptionValuation(underlying, spec, method, params=params)

    def _make_gbm(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> GBMProcess:
        params = GBMParams(
            initial_value=self.spot if spot is None else spot,
            volatility=self.volatility if vol is None else vol,
            dividend_curve=dividend_curve,
        )
        return GBMProcess(self.market_data, params, self.sim_config)


class TestDeltaBasicProperties(TestGreeksSetup):
    """Test basic properties of delta values."""

    def test_call_delta_positive(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() > 0

    def test_put_delta_negative(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() < 0

    def test_call_delta_atm_greater_than_half(self):
        """With positive r, forward > spot so ATM call delta > 0.5."""
        spec = self._make_spec(option_type=OptionType.CALL, strike=self.strike)
        ud = self._make_ud(spot=self.spot)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        delta = valuation.delta()
        assert 0.5 < delta < 1.0

    def test_call_delta_itm_close_to_one(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud(spot=150.0)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() > 0.95

    def test_call_delta_otm_close_to_zero(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud(spot=50.0)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() < 0.05

    def test_put_delta_itm_close_to_negative_one(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud(spot=50.0)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() < -0.95

    def test_put_delta_otm_close_to_zero(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud(spot=150.0)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.delta() > -0.05

    def test_delta_custom_epsilon(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)

        delta = valuation.delta(epsilon=0.5)
        assert 0 < delta < 1


class TestGammaBasicProperties(TestGreeksSetup):
    """Test basic properties of gamma values."""

    def test_gamma_always_positive(self):
        call_spec = self._make_spec(option_type=OptionType.CALL)
        put_spec = self._make_spec(option_type=OptionType.PUT)
        ud_call = self._make_ud()
        ud_put = self._make_ud()

        call_val = self._make_val(ud_call, call_spec, PricingMethod.BSM)
        put_val = self._make_val(ud_put, put_spec, PricingMethod.BSM)

        assert call_val.gamma() > 0
        assert put_val.gamma() > 0

    def test_gamma_highest_atm(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        # ATM gamma
        ud_atm = self._make_ud(spot=self.spot)
        val_atm = self._make_val(ud_atm, spec, PricingMethod.BSM)
        gamma_atm = val_atm.gamma()

        # ITM gamma (separate underlying instance)
        ud_itm = self._make_ud(spot=110.0)
        val_itm = self._make_val(ud_itm, spec, PricingMethod.BSM)
        gamma_itm = val_itm.gamma()

        assert gamma_atm > gamma_itm

    def test_gamma_decreases_with_time(self):
        maturity_short = dt.datetime(2025, 3, 1)

        ud_short = self._make_ud(pricing_date=self.pricing_date)
        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        val_short = self._make_val(ud_short, spec_short, PricingMethod.BSM)
        gamma_short = val_short.gamma()

        ud_long = self._make_ud(pricing_date=self.pricing_date)
        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)
        val_long = self._make_val(ud_long, spec_long, PricingMethod.BSM)
        gamma_long = val_long.gamma()

        assert gamma_short > gamma_long


class TestVegaBasicProperties(TestGreeksSetup):
    """Test basic properties of vega values."""

    def test_vega_always_positive(self):
        call_spec = self._make_spec(option_type=OptionType.CALL)
        put_spec = self._make_spec(option_type=OptionType.PUT)
        ud_call = self._make_ud()
        ud_put = self._make_ud()

        call_val = self._make_val(ud_call, call_spec, PricingMethod.BSM)
        put_val = self._make_val(ud_put, put_spec, PricingMethod.BSM)

        assert call_val.vega() > 0
        assert put_val.vega() > 0

    def test_vega_highest_atm(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        ud_atm = self._make_ud(spot=self.spot)
        val_atm = self._make_val(ud_atm, spec, PricingMethod.BSM)
        vega_atm = val_atm.vega()

        ud_itm = self._make_ud(spot=110.0)
        val_itm = self._make_val(ud_itm, spec, PricingMethod.BSM)
        vega_itm = val_itm.vega()

        assert vega_atm > vega_itm

    def test_vega_increases_with_time(self):
        maturity_short = dt.datetime(2025, 3, 1)
        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        ud_short = self._make_ud()
        val_short = self._make_val(ud_short, spec_short, PricingMethod.BSM)
        vega_short = val_short.vega()

        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)
        ud_long = self._make_ud()
        val_long = self._make_val(ud_long, spec_long, PricingMethod.BSM)
        vega_long = val_long.vega()

        assert vega_long > vega_short

    def test_vega_custom_epsilon(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)

        vega = valuation.vega(epsilon=0.02)
        assert vega > 0


class TestRhoBasicProperties(TestGreeksSetup):
    """Test basic properties of rho values."""

    def test_call_rho_positive(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(self._make_ud(), spec, PricingMethod.BSM)
        assert valuation.rho() > 0

    def test_put_rho_negative(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        valuation = self._make_val(self._make_ud(), spec, PricingMethod.BSM)
        assert valuation.rho() < 0

    def test_rho_increases_with_maturity(self):
        maturity_short = dt.datetime(2025, 4, 1)

        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)

        rho_short = self._make_val(self._make_ud(), spec_short, PricingMethod.BSM).rho()
        rho_long = self._make_val(self._make_ud(), spec_long, PricingMethod.BSM).rho()

        assert rho_long > rho_short > 0

    def test_numerical_rho_close_to_analytical(self):
        valuation = self._make_val(
            self._make_ud(),
            self._make_spec(option_type=OptionType.CALL),
            PricingMethod.BSM,
        )
        rho_analytical = valuation.rho(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        rho_numerical = valuation.rho(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(rho_analytical, rho_numerical, rtol=0.01)

    def test_itm_call_rho_greater_than_otm(self):
        rho_itm = self._make_val(
            self._make_ud(),
            self._make_spec(option_type=OptionType.CALL, strike=80.0),
            PricingMethod.BSM,
        ).rho()
        rho_otm = self._make_val(
            self._make_ud(),
            self._make_spec(option_type=OptionType.CALL, strike=120.0),
            PricingMethod.BSM,
        ).rho()
        assert rho_itm > rho_otm


class TestThetaBasicProperties(TestGreeksSetup):
    """Test basic properties of theta values."""

    def test_call_theta_negative(self):
        """ATM European call theta should be negative (time decay)."""
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.theta() < 0

    def test_put_theta_negative(self):
        """ATM European put theta should be negative (no dividends, moderate rate)."""
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        assert valuation.theta() < 0

    def test_theta_magnitude_increases_near_expiry(self):
        """ATM theta should become more negative (larger magnitude) closer to expiry."""
        maturity_short = dt.datetime(2025, 2, 1)
        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        ud_short = self._make_ud()
        val_short = self._make_val(ud_short, spec_short, PricingMethod.BSM)
        theta_short = val_short.theta()

        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)
        ud_long = self._make_ud()
        val_long = self._make_val(ud_long, spec_long, PricingMethod.BSM)
        theta_long = val_long.theta()

        # Shorter maturity → larger magnitude theta (more negative)
        assert theta_short < theta_long < 0

    def test_theta_atm_highest_magnitude(self):
        """ATM options have the largest theta magnitude."""
        spec = self._make_spec(option_type=OptionType.CALL)

        ud_atm = self._make_ud(spot=self.spot)
        val_atm = self._make_val(ud_atm, spec, PricingMethod.BSM)
        theta_atm = val_atm.theta()

        ud_otm = self._make_ud(spot=70.0)
        val_otm = self._make_val(ud_otm, spec, PricingMethod.BSM)
        theta_otm = val_otm.theta()

        # ATM theta more negative than OTM theta
        assert theta_atm < theta_otm

    def test_theta_with_dividends_call(self):
        """Continuous dividend yield affects call theta."""
        spec = self._make_spec(option_type=OptionType.CALL)

        val_no_div = self._make_val(
            self._make_ud(dividend_curve=None),
            spec,
            PricingMethod.BSM,
        )
        theta_no_div = val_no_div.theta()

        val_with_div = self._make_val(
            self._make_ud(dividend_curve=self._make_q_curve(0.03)),
            spec,
            PricingMethod.BSM,
        )
        theta_with_div = val_with_div.theta()

        # Both negative, values differ due to dividend
        assert theta_no_div < 0
        assert theta_with_div < 0
        assert not np.isclose(theta_no_div, theta_with_div)

    def test_theta_near_expiry_large_magnitude(self):
        """Theta magnitude should be very large for near-expiry ATM options."""
        maturity_near = self.pricing_date + dt.timedelta(days=2)
        spec = self._make_spec(option_type=OptionType.CALL, maturity=maturity_near)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)
        theta = valuation.theta()
        # Near-expiry ATM call has very large negative theta
        assert theta < -0.05


class TestGreekCalculationMethods(TestGreeksSetup):
    """Light consistency checks between analytical and numerical BSM Greeks.

    Broad multi-scenario benchmarking is covered in test_quantlib_greeks_comparison.py.
    """

    @pytest.mark.parametrize(
        "option_type,greek,rtol",
        [
            (OptionType.CALL, "delta", 1e-3),
            (OptionType.CALL, "gamma", 1e-3),
            (OptionType.CALL, "vega", 1e-3),
            (OptionType.CALL, "theta", 0.02),
            (OptionType.PUT, "theta", 0.02),
        ],
    )
    def test_bsm_analytical_vs_numerical_smoke(self, option_type, greek, rtol):
        spec = self._make_spec(option_type=option_type)
        ud = self._make_ud()
        valuation = self._make_val(ud, spec, PricingMethod.BSM)

        analytic_fn = getattr(valuation, greek)
        analytical = analytic_fn(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        numerical = analytic_fn(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(analytical, numerical, rtol=rtol)

    def test_binomial_tree_theta_matches_numerical_with_comparable_time_bump(self):
        """Tree theta and bump-and-revalue theta should align when bump uses 2*dt.

        Binomial tree theta uses (f_ud - f_0) / (2*dt), so a numerical theta
        bump of approximately 2*dt days is the comparable finite-difference
        horizon.
        """
        num_steps = 200
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=num_steps),
        )

        ttm = calculate_year_fraction(
            self.pricing_date,
            self.maturity,
            day_count_convention=valuation.day_count_convention,
        )

        dt = ttm / num_steps

        # Binomial tree theta uses a 2*dt horizon in year units, then reports
        # per calendar day (division by 365). Use the same horizon here.
        time_bump_days = 2.0 * dt * 365.0

        theta_tree = valuation.theta(greek_calc_method=GreekCalculationMethod.TREE)
        theta_num = valuation.theta(
            greek_calc_method=GreekCalculationMethod.NUMERICAL,
            time_bump_days=time_bump_days,
        )

        assert np.isclose(theta_tree, theta_num, rtol=2e-3, atol=5e-5)


class TestGreeksDividendCurveEffect(TestGreeksSetup):
    """Test the effect of dividend curves on greeks."""

    def test_call_delta_lower_with_dividend_curve(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        val_no_div = self._make_val(
            self._make_ud(dividend_curve=None),
            spec,
            PricingMethod.BSM,
        )
        delta_no_div = val_no_div.delta()

        val_with_div = self._make_val(
            self._make_ud(dividend_curve=self._make_q_curve(0.03)),
            spec,
            PricingMethod.BSM,
        )
        delta_with_div = val_with_div.delta()

        assert delta_with_div < delta_no_div

    def test_put_delta_more_negative_with_dividend_curve(self):
        spec = self._make_spec(option_type=OptionType.PUT)

        val_no_div = self._make_val(
            self._make_ud(dividend_curve=None),
            spec,
            PricingMethod.BSM,
        )
        delta_no_div = val_no_div.delta()

        val_with_div = self._make_val(
            self._make_ud(dividend_curve=self._make_q_curve(0.03)),
            spec,
            PricingMethod.BSM,
        )
        delta_with_div = val_with_div.delta()

        assert delta_with_div < delta_no_div


class TestGreekErrorHandling(TestGreeksSetup):
    """Test error handling for greek calculations."""

    def test_analytical_greek_with_non_bsm_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        valuation = self._make_val(self._make_ud(), spec, PricingMethod.BINOMIAL)

        with pytest.raises(
            ValidationError, match=r"Analytical greeks are only available for BSM.*"
        ):
            valuation.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_tree_greek_with_non_binomial_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(self._make_ud(), spec, PricingMethod.BSM)

        with pytest.raises(ValidationError, match=r"Tree greeks are only available for BINOMIAL.*"):
            valuation.delta(greek_calc_method=GreekCalculationMethod.TREE)

    def test_tree_greek_for_vega_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(
            self._make_ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        with pytest.raises(ValidationError, match="Tree extraction is not available"):
            valuation.vega(greek_calc_method=GreekCalculationMethod.TREE)

    def test_invalid_greek_calc_method_type_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        valuation = self._make_val(self._make_ud(), spec, PricingMethod.BSM)

        with pytest.raises(
            ConfigurationError, match="greek_calc_method must be GreekCalculationMethod enum"
        ):
            valuation.delta(greek_calc_method="analytical")  # type: ignore[arg-type]


class TestGreekImmutability(TestGreeksSetup):
    """Test that greek calculations don't mutate underlying state (thread-safety)."""

    def test_delta_does_not_mutate_underlying_pricing_data(self):
        """Verify delta calculation doesn't mutate UnderlyingData.initial_value."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate delta with numerical method
        delta = valuation.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Delta calculation mutated initial_value"
        assert ud.volatility == original_vol, "Delta calculation mutated volatility"
        assert delta is not None  # sanity check

    def test_gamma_does_not_mutate_underlying_pricing_data(self):
        """Verify gamma calculation doesn't mutate UnderlyingData.initial_value."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate gamma
        gamma = valuation.gamma()

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Gamma calculation mutated initial_value"
        assert ud.volatility == original_vol, "Gamma calculation mutated volatility"
        assert gamma is not None  # sanity check

    def test_vega_does_not_mutate_underlying_pricing_data(self):
        """Verify vega calculation doesn't mutate UnderlyingData.volatility."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(ud, spec, PricingMethod.BSM)

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate vega with numerical method
        vega = valuation.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Vega calculation mutated initial_value"
        assert ud.volatility == original_vol, "Vega calculation mutated volatility"
        assert vega is not None  # sanity check

    def test_delta_does_not_mutate_path_simulation(self):
        """Verify delta calculation doesn't mutate PathSimulation.initial_value."""
        # Create PathSimulation
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=self.spot, volatility=self.volatility),
            sim_config=SimulationConfig(
                paths=1000,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = OptionValuation(
            underlying=process,
            spec=spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )

        original_spot = process.initial_value
        original_vol = process.volatility

        # Calculate delta
        delta = valuation.delta()

        # Verify PathSimulation state is unchanged
        assert process.initial_value == original_spot, (
            "Delta calculation mutated PathSimulation.initial_value"
        )
        assert process.volatility == original_vol, (
            "Delta calculation mutated PathSimulation.volatility"
        )
        assert delta is not None  # sanity check

    def test_multiple_concurrent_greeks_with_shared_underlying(self):
        """Verify multiple valuations can share underlying without interference.

        This simulates thread-safety scenario where multiple portfolios
        might calculate greeks for the same underlying simultaneously.
        """
        # Shared underlying
        ud = self._make_ud()
        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Create multiple valuations with different strikes sharing same underlying
        spec1 = self._make_spec(option_type=OptionType.CALL, strike=90.0)
        spec2 = self._make_spec(option_type=OptionType.CALL, strike=100.0)
        spec3 = self._make_spec(option_type=OptionType.PUT, strike=110.0)

        val1 = self._make_val(ud, spec1, PricingMethod.BSM)
        val2 = self._make_val(ud, spec2, PricingMethod.BSM)
        val3 = self._make_val(ud, spec3, PricingMethod.BSM)

        # Calculate greeks for all valuations (simulating concurrent access)
        delta1 = val1.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        gamma2 = val2.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        vega3 = val3.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Shared underlying was mutated"
        assert ud.volatility == original_vol, "Shared underlying volatility was mutated"

        # Verify results are reasonable
        assert 0 < delta1 < 1.0  # ITM call delta
        assert gamma2 > 0  # ATM gamma is positive
        assert vega3 > 0  # vega always positive


# ═══════════════════════════════════════════════════════════════════════
# Asian Greeks not covered by QuantLib comparisons
# ═══════════════════════════════════════════════════════════════════════

_PRICING_DATE = dt.datetime(2025, 1, 1)
_MATURITY = dt.datetime(2026, 1, 1)
_CURRENCY = "USD"
_VOL = 0.20
_MC_SEED = 42
_ASIAN_OBSERVATIONS = 52
_ASIAN_ENGINE_STEPS = _ASIAN_OBSERVATIONS - 1
_ASIAN_THETA_BUMP_DAYS = 7

_NONFLAT_R = DiscountCurve.from_forwards(
    times=np.array([0.0, 0.25, 0.5, 1.0]),
    forwards=np.array([0.03, 0.05, 0.06]),
)
_NONFLAT_Q = DiscountCurve.from_forwards(
    times=np.array([0.0, 0.25, 0.5, 1.0]),
    forwards=np.array([0.01, 0.02, 0.015]),
)
_DISCRETE_DIVS = [
    (_PRICING_DATE + dt.timedelta(days=90), 0.50),
    (_PRICING_DATE + dt.timedelta(days=270), 0.50),
]


def _asian_curve(kind: str, *, is_dividend: bool) -> DiscountCurve | None:
    if kind == "none":
        return None
    if kind == "flat":
        rate = 0.03 if is_dividend else 0.05
        return flat_curve(_PRICING_DATE, _MATURITY, rate)
    if kind == "nonflat":
        return _NONFLAT_Q if is_dividend else _NONFLAT_R
    raise ValueError(f"unsupported curve kind: {kind}")


def _asian_md(rate_curve: DiscountCurve) -> MarketData:
    return MarketData(_PRICING_DATE, rate_curve, currency=_CURRENCY)


def _asian_ud(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    averaging: AsianAveraging,
    rate_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    discrete_dividends: list[tuple[dt.datetime, float]] | None,
) -> tuple[UnderlyingData, AsianSpec]:
    ud = UnderlyingData(
        initial_value=spot,
        volatility=_VOL,
        market_data=_asian_md(rate_curve),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=_MATURITY,
        currency=_CURRENCY,
        exercise_type=ExerciseType.AMERICAN,
        num_observations=_ASIAN_OBSERVATIONS,
    )
    return ud, spec


def _asian_mc(
    *,
    spot: float,
    rate_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    discrete_dividends: list[tuple[dt.datetime, float]] | None,
) -> GBMProcess:
    return GBMProcess(
        _asian_md(rate_curve),
        GBMParams(
            initial_value=spot,
            volatility=_VOL,
            dividend_curve=dividend_curve,
            discrete_dividends=discrete_dividends,
        ),
        SimulationConfig(paths=120_000, num_steps=_ASIAN_ENGINE_STEPS, end_date=_MATURITY),
    )


def _greeks(ov: OptionValuation) -> dict[str, float]:
    return {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        # A 1-day theta bump is too noisy for MC in these Asian AM scenarios;
        # use a wider bump so binomial-vs-MC comparison is signal-dominated.
        "theta": ov.theta(
            greek_calc_method=GreekCalculationMethod.NUMERICAL,
            time_bump_days=_ASIAN_THETA_BUMP_DAYS,
        ),
        "rho": ov.rho(),
    }


_ASIAN_AM_DP_SCENARIOS = [
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        AsianAveraging.GEOMETRIC,
        "flat",
        "none",
        None,
        id="am_geom_put_atm_flat",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        "flat",
        "flat",
        None,
        id="am_arith_put_atm_flat_div",
    ),
    pytest.param(
        95.0,
        105.0,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        "nonflat",
        "nonflat",
        None,
        id="am_arith_call_otm_nonflat",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        "nonflat",
        "none",
        _DISCRETE_DIVS,
        id="am_arith_put_atm_nonflat_discrete",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "spot,strike,option_type,averaging,rate_kind,div_kind,discrete_dividends",
    _ASIAN_AM_DP_SCENARIOS,
)
def test_asian_american_greeks_binomial_vs_mc(
    spot,
    strike,
    option_type,
    averaging,
    rate_kind,
    div_kind,
    discrete_dividends,
):
    """For Asian scenarios unsupported by QL, DP binomial and MC greeks should broadly align."""
    r_curve = _asian_curve(rate_kind, is_dividend=False)
    q_curve = _asian_curve(div_kind, is_dividend=True)
    assert r_curve is not None

    ud, spec = _asian_ud(
        spot=spot,
        strike=strike,
        option_type=option_type,
        averaging=averaging,
        rate_curve=r_curve,
        dividend_curve=q_curve,
        discrete_dividends=discrete_dividends,
    )
    mc_process = _asian_mc(
        spot=spot,
        rate_curve=r_curve,
        dividend_curve=q_curve,
        discrete_dividends=discrete_dividends,
    )

    # For Hull-style Asian trees, use representative averages ~= 2x num_steps.
    binom = OptionValuation(
        ud,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=_ASIAN_ENGINE_STEPS,
            asian_tree_averages=2 * _ASIAN_ENGINE_STEPS,
        ),
    )
    mc = OptionValuation(
        mc_process,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_MC_SEED),
    )

    g_bin = _greeks(binom)
    g_mc = _greeks(mc)
    tols = {"delta": 0.15, "gamma": 0.30, "vega": 0.20, "theta": 0.30, "rho": 0.20}

    assert_greeks_close(
        lhs=g_bin,
        rhs=g_mc,
        tols=tols,
        log_prefix=(
            f"Asian AM {averaging.value} {option_type.value} "
            f"r={rate_kind} q={div_kind} disc_div={discrete_dividends is not None}"
        ),
        lhs_name="Binom",
        rhs_name="MC",
        atol=1e-3,
        logger=logger,
    )


# ── Asian auto-select Greek method → NUMERICAL ─────────────────────────


class TestAsianGreekMethodSelection(TestGreeksSetup):
    """Verify Asian specs always use NUMERICAL Greek method."""

    def test_asian_bsm_auto_selects_numerical(self):
        """BSM Asian should auto-select NUMERICAL, not ANALYTICAL."""
        ud = self._make_ud()
        spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.CALL,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
            num_observations=12,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(ud, spec, PricingMethod.BSM)
        # Should not raise — would crash with AttributeError if ANALYTICAL were used
        delta = ov.delta()
        assert np.isfinite(delta)

    def test_asian_binomial_auto_selects_numerical(self):
        """Binomial Asian should auto-select NUMERICAL, not TREE."""
        ud = self._make_ud()
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
            num_observations=12,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100, asian_tree_averages=150),
        )
        delta = ov.delta()
        assert np.isfinite(delta)

    def test_asian_explicit_analytical_raises(self):
        """Explicitly requesting ANALYTICAL for Asian should raise."""
        ud = self._make_ud()
        spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.CALL,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
            num_observations=12,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(ud, spec, PricingMethod.BSM)
        with pytest.raises(UnsupportedFeatureError, match="Asian options only support.*NUMERICAL"):
            ov.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_asian_explicit_tree_raises(self):
        """Explicitly requesting TREE for Asian should raise."""
        ud = self._make_ud()
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
            num_observations=12,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100, asian_tree_averages=150),
        )
        with pytest.raises(UnsupportedFeatureError, match="Asian options only support.*NUMERICAL"):
            ov.delta(greek_calc_method=GreekCalculationMethod.TREE)


# ═══════════════════════════════════════════════════════════════════════
# Barrier Greeks
# ═══════════════════════════════════════════════════════════════════════
PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"

_BARRIER_SPOT = 100.0
_BARRIER_VOL = 0.25
_BARRIER_RATE = 0.05
_BARRIER_DIV = 0.02
_BARRIER_NUMERICAL_SPOT_BUMP_RATIO = 0.025
_BARRIER_NUMERICAL_THETA_DAYS = 21.0
_BARRIER_PDE_CFG = PDEParams(
    spot_steps=800,
    time_steps=800,
    method=PDEMethod.CRANK_NICOLSON,
    space_grid=PDESpaceGrid.LOG_SPOT,
)
_BARRIER_BINOM_CFG = BinomialParams(num_steps=400)


def _barrier_numerical_spot_bump(spot: float) -> float:
    return float(spot) * _BARRIER_NUMERICAL_SPOT_BUMP_RATIO


_ALL_BARRIER_GREEKS: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")


def _dp_barrier_greeks_from_valuation(
    ov: OptionValuation,
    *,
    spot: float,
    greeks: tuple[str, ...] = _ALL_BARRIER_GREEKS,
) -> dict[str, float]:
    """Compute the requested barrier greeks for a valuation.

    Only the greeks named in ``greeks`` are computed, so callers that
    need a subset (e.g. just rho, or just vega/rho) avoid the compute
    cost of the others.  Binomial barrier valuations must omit "vega"
    because there is no tree-native vega and NUMERICAL bump-and-revalue
    is blocked on binomial barriers (Boyle-Lau retopologises the tree on
    each vol bump).
    """
    bump = _barrier_numerical_spot_bump(spot)
    result: dict[str, float] = {}
    if "delta" in greeks:
        result["delta"] = ov.delta(epsilon=bump)
    if "gamma" in greeks:
        result["gamma"] = ov.gamma(epsilon=bump)
    if "vega" in greeks:
        result["vega"] = ov.vega()
    if "theta" in greeks:
        result["theta"] = ov.theta(time_bump_days=_BARRIER_NUMERICAL_THETA_DAYS)
    if "rho" in greeks:
        result["rho"] = ov.rho()
    return result


def _fmt_greek_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _dp_barrier_greeks(
    *,
    pricing_method: PricingMethod,
    params: BinomialParams | PDEParams | None = None,
    spot: float = _BARRIER_SPOT,
    volatility: float = _BARRIER_VOL,
    exercise_type: ExerciseType,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
    greeks: tuple[str, ...] = _ALL_BARRIER_GREEKS,
) -> dict[str, float]:
    """Compute the requested barrier Greeks via a selected DP pricing engine."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    rc = r_curve if r_curve is not None else DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    qc = q_curve if q_curve is not None else DiscountCurve.flat(_BARRIER_DIV, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    ud = UnderlyingData(
        initial_value=spot,
        volatility=volatility,
        market_data=md,
        dividend_curve=qc,
    )
    spec = BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    ov = OptionValuation(ud, spec, pricing_method, params=params)
    return _dp_barrier_greeks_from_valuation(ov, spot=spot, greeks=greeks)


def _make_barrier_valuation(
    *,
    pricing_method: PricingMethod,
    params: BinomialParams | PDEParams | None = None,
    spot: float = _BARRIER_SPOT,
    volatility: float = _BARRIER_VOL,
    exercise_type: ExerciseType,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> OptionValuation:
    """Build a barrier valuation with the shared test-market defaults."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    rc = r_curve if r_curve is not None else DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    qc = q_curve if q_curve is not None else DiscountCurve.flat(_BARRIER_DIV, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    ud = UnderlyingData(
        initial_value=spot,
        volatility=volatility,
        market_data=md,
        dividend_curve=qc,
    )
    spec = BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    return OptionValuation(ud, spec, pricing_method, params=params)


def _barrier_nonflat_curves() -> tuple[DiscountCurve, DiscountCurve]:
    """Non-flat rate and dividend curves for barrier tests."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    r_times = np.array([0.0, 0.25, 0.5, ttm])
    r_forwards = np.array([0.03, 0.06, 0.04])
    q_times = np.array([0.0, 0.25, 0.5, ttm])
    q_forwards = np.array([0.01, 0.03, 0.005])
    return (
        DiscountCurve.from_forwards(times=r_times, forwards=r_forwards),
        DiscountCurve.from_forwards(times=q_times, forwards=q_forwards),
    )


_BARRIER_AMERICAN_NONFLAT_GREEK_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        105.0,
        90.0,
        0.0,
        RebateTiming.AT_HIT,
        *_barrier_nonflat_curves(),
        id="am_down_out_put_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_HIT,
        *_barrier_nonflat_curves(),
        id="am_up_out_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_HIT,
        *_barrier_nonflat_curves(),
        id="am_down_out_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        95.0,
        110.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        *_barrier_nonflat_curves(),
        id="am_up_in_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        95.0,
        80.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        *_barrier_nonflat_curves(),
        id="am_down_in_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        110.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        *_barrier_nonflat_curves(),
        id="am_up_in_put_nonflat",
    ),
]


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing,r_curve,q_curve",
    _BARRIER_AMERICAN_NONFLAT_GREEK_SCENARIOS,
)
def test_american_barrier_nonflat_greeks_binomial_vs_pde(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
    rebate_timing,
    r_curve,
    q_curve,
):
    """American non-flat barrier Greeks: compare PDE FD to binomial only.

    QuantLib's American barrier support is binomial-only and only approximates
    non-flat term structures, so these scenarios are validated internally using
    DP binomial as the baseline reference.
    """
    # Vega is excluded on both sides: binomial barrier vega is blocked
    # (no tree-native path; NUMERICAL guard catches Boyle-Lau retopology),
    # so there is no cross-engine counterpart to compare PDE vega against.
    nonflat_greeks = ("delta", "gamma", "theta", "rho")
    dp_bn = _dp_barrier_greeks(
        pricing_method=PricingMethod.BINOMIAL,
        params=_BARRIER_BINOM_CFG,
        exercise_type=ExerciseType.AMERICAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
        greeks=nonflat_greeks,
    )
    dp_pde = _dp_barrier_greeks(
        pricing_method=PricingMethod.PDE_FD,
        params=_BARRIER_PDE_CFG,
        exercise_type=ExerciseType.AMERICAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
        greeks=nonflat_greeks,
    )
    # DOWN-OUT put / UP-OUT call: the KO barrier truncates the payoff
    # region, amplifying CRR tree bias vs the FD grid.  Residuals up to ~12%
    # on theta/rho are genuine and converge as binomial steps increase.
    # All other scenarios agree to within ~1.1%.

    reverse_barrier_ko = (
        direction is BarrierDirection.DOWN
        and action is BarrierAction.OUT
        and option_type is OptionType.PUT
    ) or (
        direction is BarrierDirection.UP
        and action is BarrierAction.OUT
        and option_type is OptionType.CALL
    )
    bn_tols = (
        {"delta": 0.05, "gamma": 0.12, "theta": 0.15, "rho": 0.15}
        if reverse_barrier_ko
        else {"delta": 0.015, "gamma": 0.015, "theta": 0.015, "rho": 0.015}
    )

    for greek in ("delta", "gamma", "theta", "rho"):
        logger.info(
            "American barrier nonflat %s-%s %s %s K=%.0f H=%.0f | DP_BN=%s DP_PDE=%s",
            direction.value,
            action.value,
            option_type.value,
            greek,
            strike,
            barrier,
            _fmt_greek_value(dp_bn[greek]),
            _fmt_greek_value(dp_pde[greek]),
        )

    assert_greeks_close(
        lhs=dp_bn,
        rhs=dp_pde,
        tols=bn_tols,
        log_prefix=(
            f"American barrier nonflat BN {direction.value}-{action.value} {option_type.value} "
            f"K={strike:.0f} H={barrier:.0f}"
        ),
        lhs_name="DP_BN",
        rhs_name="DP_PDE",
        skip_missing_rhs=False,
        atol=1e-3,
        logger=None,
    )


@pytest.mark.parametrize("exercise_type", [ExerciseType.EUROPEAN, ExerciseType.AMERICAN])
def test_knock_out_triggered_at_inception_default_greeks_zero_without_rebate(exercise_type):
    valuation = _make_barrier_valuation(
        pricing_method=PricingMethod.BINOMIAL,
        params=_BARRIER_BINOM_CFG,
        spot=80.0,
        exercise_type=exercise_type,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.OUT,
        barrier=90.0,
        option_type=OptionType.PUT,
        strike=100.0,
        rebate=0.0,
        rebate_timing=RebateTiming.AT_HIT,
    )

    assert np.isclose(valuation.present_value(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.delta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.gamma(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.theta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.rho(), 0.0, atol=1.0e-12)


@pytest.mark.parametrize("exercise_type", [ExerciseType.EUROPEAN, ExerciseType.AMERICAN])
def test_knock_out_triggered_at_inception_default_greeks_match_fixed_expiry_rebate(exercise_type):
    rebate = 5.0
    valuation = _make_barrier_valuation(
        pricing_method=PricingMethod.BINOMIAL,
        params=_BARRIER_BINOM_CFG,
        spot=80.0,
        exercise_type=exercise_type,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.OUT,
        barrier=90.0,
        option_type=OptionType.PUT,
        strike=100.0,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )

    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    discount_factor = float(DiscountCurve.flat(_BARRIER_RATE, end_time=ttm).df(ttm))
    expected_pv = rebate * discount_factor
    expected_theta = rebate * discount_factor * _BARRIER_RATE / 365.0
    expected_rho = -expected_pv * ttm * 0.01

    assert np.isclose(valuation.present_value(), expected_pv, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(valuation.delta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.gamma(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.theta(), expected_theta, rtol=0.02, atol=1.0e-6)
    assert np.isclose(valuation.rho(), expected_rho, rtol=0.02, atol=1.0e-6)


# ═══════════════════════════════════════════════════════════════════════
# Barrier vega / rho — cross-engine validation
# ═══════════════════════════════════════════════════════════════════════
# QuantLib's barrier engines (Analytic/FD/Binomial) don't expose vega/rho,
# so we validate DP's numerical vega/rho with an internal cross-check.
#
# European: BSM (closed-form barrier formulas, NUMERICAL bump-and-revalue)
#   vs PDE_FD (NUMERICAL).  Binomial is NOT used for the vega cross-check
#   because the binomial barrier engine blocks NUMERICAL vega: bumping vol
#   retopologises the Boyle-Lau-aligned tree so the central difference
#   compares two different trees.
#
# American: only rho is cross-checked (binomial vs PDE) — rho is exempt
#   from the binomial-barrier NUMERICAL guard because rate bumps don't
#   enter the Boyle-Lau alignment formula.  American vega has no cross-
#   engine counterpart: BSM is European-only, binomial is blocked.

_BARRIER_VEGA_RHO_EU_FLAT_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        id="eu_down_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        105.0,
        115.0,
        id="eu_up_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        id="eu_down_in_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        95.0,
        125.0,
        id="eu_up_out_put_flat",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier",
    _BARRIER_VEGA_RHO_EU_FLAT_SCENARIOS,
)
def test_european_barrier_vega_rho_bsm_vs_pde(direction, action, option_type, strike, barrier):
    """European flat barrier vega/rho: DP BSM vs DP PDE_FD cross-check.

    QL barrier engines don't expose vega/rho, so these scenarios (matching
    test_quantlib_greeks_comparison._BARRIER_GREEK_SCENARIOS flat cases) are
    validated via BSM (NUMERICAL bump of closed-form barrier formulas) vs
    PDE_FD (NUMERICAL bump of the full FD grid).  Binomial is deliberately
    excluded: bumping vol retopologises the Boyle-Lau-aligned tree and the
    central difference becomes invalid.
    """
    vega_rho = ("vega", "rho")
    dp_bsm = _dp_barrier_greeks(
        pricing_method=PricingMethod.BSM,
        exercise_type=ExerciseType.EUROPEAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        greeks=vega_rho,
    )
    dp_pde = _dp_barrier_greeks(
        pricing_method=PricingMethod.PDE_FD,
        params=_BARRIER_PDE_CFG,
        exercise_type=ExerciseType.EUROPEAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        greeks=vega_rho,
    )

    tols = {"vega": 0.02, "rho": 0.02}
    assert_greeks_close(
        lhs=dp_pde,
        rhs=dp_bsm,
        tols=tols,
        log_prefix=(
            f"European barrier vega/rho {direction.value}-{action.value} "
            f"{option_type.value} K={strike:.0f} H={barrier:.0f}"
        ),
        lhs_name="DP_PDE",
        rhs_name="DP_BSM",
        skip_missing_rhs=False,
        atol=1e-3,
        logger=None,
    )


_BARRIER_RHO_AM_FLAT_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_HIT,
        id="am_down_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_HIT,
        id="am_down_out_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_HIT,
        id="am_up_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_HIT,
        id="am_up_out_put_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        RebateTiming.AT_HIT,
        id="am_down_out_call_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_HIT,
        id="am_up_out_put_rebate",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="am_down_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="am_down_in_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="am_up_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="am_up_in_put_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        RebateTiming.AT_EXPIRY,
        id="am_down_in_call_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_EXPIRY,
        id="am_up_in_put_rebate",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing",
    _BARRIER_RHO_AM_FLAT_SCENARIOS,
)
def test_american_barrier_rho_binomial_vs_pde(
    direction, action, option_type, strike, barrier, rebate, rebate_timing
):
    """American flat barrier rho: DP binomial vs DP PDE_FD cross-check.

    Mirrors test_quantlib_greeks_comparison._BARRIER_AMERICAN_GREEK_SCENARIOS
    for coverage of rho which QL's barrier engines do not expose. Vega is
    not cross-checked for American barriers: binomial barrier vega is
    blocked by the NUMERICAL guard (rate bumps are exempt; vol bumps are
    not), and BSM has no American support.
    """
    rho_only = ("rho",)
    dp_bn = _dp_barrier_greeks(
        pricing_method=PricingMethod.BINOMIAL,
        params=_BARRIER_BINOM_CFG,
        exercise_type=ExerciseType.AMERICAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        greeks=rho_only,
    )
    dp_pde = _dp_barrier_greeks(
        pricing_method=PricingMethod.PDE_FD,
        params=_BARRIER_PDE_CFG,
        exercise_type=ExerciseType.AMERICAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        greeks=rho_only,
    )

    reverse_barrier_ko = (
        direction is BarrierDirection.DOWN
        and action is BarrierAction.OUT
        and option_type is OptionType.PUT
    ) or (
        direction is BarrierDirection.UP
        and action is BarrierAction.OUT
        and option_type is OptionType.CALL
    )

    rtol = {"rho": 0.015} if not reverse_barrier_ko else {"rho": 0.1}
    assert_greeks_close(
        lhs=dp_pde,
        rhs=dp_bn,
        tols=rtol,
        log_prefix=(
            f"American barrier rho {direction.value}-{action.value} "
            f"{option_type.value} K={strike:.0f} H={barrier:.0f} R={rebate:.1f}"
        ),
        lhs_name="DP_PDE",
        rhs_name="DP_BN",
        skip_missing_rhs=False,
        atol=1e-3,
        logger=None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Barrier Monte Carlo greeks
# ═══════════════════════════════════════════════════════════════════════
# European MC barrier greeks vs BSM analytical (BSM is the ground truth).
# American MC barrier greeks vs PDE_FD (PDE is the ground truth).
# Tolerances are loose because MC bump-and-revalue accumulates Monte Carlo
# variance on top of finite-difference noise; barrier payoffs amplify both.

_BARRIER_MC_PATHS = 150_000
_BARRIER_MC_STEPS = 200
_BARRIER_MC_CFG = MonteCarloParams(random_seed=42, deg=3, barrier_aware_basis=True)


def _dp_barrier_mc_greeks(
    *,
    exercise_type: ExerciseType,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    greeks: tuple[str, ...] = _ALL_BARRIER_GREEKS,
) -> dict[str, float]:
    """Compute requested barrier greeks via Monte Carlo (GBMProcess underlying)."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    rc = DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    qc = DiscountCurve.flat(_BARRIER_DIV, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    gbm = GBMProcess(
        md,
        GBMParams(
            initial_value=_BARRIER_SPOT,
            volatility=_BARRIER_VOL,
            dividend_curve=qc,
        ),
        SimulationConfig(
            paths=_BARRIER_MC_PATHS,
            num_steps=_BARRIER_MC_STEPS,
            end_date=MATURITY,
        ),
    )
    spec = BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    ov = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=_BARRIER_MC_CFG)
    return _dp_barrier_greeks_from_valuation(ov, spot=_BARRIER_SPOT, greeks=greeks)


# Avoid the two LSM downward-bias scenarios (DOWN-OUT put, UP-OUT call) that
# would need much wider tolerances; they're independently covered by the
# American KO PV tests in test_quantlib_comparison.
_BARRIER_MC_EU_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        id="eu_down_out_call",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        id="eu_down_in_put",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        id="eu_up_out_put",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        120.0,
        id="eu_up_in_call",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier",
    _BARRIER_MC_EU_SCENARIOS,
)
def test_european_barrier_mc_greeks_vs_bsm(direction, action, option_type, strike, barrier):
    """European barrier MC greeks vs BSM analytical (closed-form ground truth).

    BSM here uses NUMERICAL bump-and-revalue on closed-form barrier formulas,
    which is essentially exact. MC uses NUMERICAL bump-and-revalue on
    simulated paths with a fixed seed.
    """
    dp_bsm = _dp_barrier_greeks(
        pricing_method=PricingMethod.BSM,
        exercise_type=ExerciseType.EUROPEAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
    )
    dp_mc = _dp_barrier_mc_greeks(
        exercise_type=ExerciseType.EUROPEAN,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
    )

    # MC bump-and-revalue noise on barrier payoffs is substantial; loose tols.
    tols = {"delta": 0.05, "gamma": 0.20, "vega": 0.10, "theta": 0.10, "rho": 0.05}
    assert_greeks_close(
        lhs=dp_mc,
        rhs=dp_bsm,
        tols=tols,
        log_prefix=(
            f"European barrier MC {direction.value}-{action.value} "
            f"{option_type.value} K={strike:.0f} H={barrier:.0f}"
        ),
        lhs_name="DP_MC",
        rhs_name="DP_BSM",
        skip_missing_rhs=False,
        atol=5e-3,
        logger=None,
    )
