"""Comprehensive tests for valuation dispatcher, specs, and integration."""

import pytest
import datetime as dt
import numpy as np
from derivatives_pricing.exceptions import (
    ConfigurationError,
    UnsupportedFeatureError,
    ValidationError,
)
from derivatives_pricing.valuation import (
    VanillaSpec,
    PayoffSpec,
    AsianSpec,
    UnderlyingData,
    OptionValuation,
    BinomialParams,
    MonteCarloParams,
    PDEParams,
    as_underlying_data,
)
from derivatives_pricing.valuation.bsm import _BSMEuropeanValuation
from derivatives_pricing.valuation.binomial import (
    _BinomialEuropeanValuation,
)
from derivatives_pricing.valuation.monte_carlo import _MCEuropeanValuation
from derivatives_pricing.enums import (
    AsianAveraging,
    DayCountConvention,
    OptionType,
    ExerciseType,
    PricingMethod,
)
from derivatives_pricing.utils import calculate_year_fraction
from derivatives_pricing.stochastic_processes import (
    GBMProcess,
    GBMParams,
    JDProcess,
    JDParams,
    SRDProcess,
    SRDParams,
    SimulationConfig,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation.pde import _FDAmericanValuation
from helpers import flat_curve


class TestVanillaSpec:
    """Tests for VanillaSpec dataclass."""

    def test_valid_option_spec_creation(self):
        """Test successful creation of valid VanillaSpec."""
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
            contract_size=100,
        )
        assert spec.option_type == OptionType.CALL
        assert spec.exercise_type == ExerciseType.EUROPEAN
        assert spec.strike == 100.0
        assert spec.contract_size == 100

    def test_option_spec_with_none_strike(self):
        """Test that None strike is rejected for vanilla VanillaSpec."""
        with pytest.raises(ValidationError, match="VanillaSpec\\.strike must be provided"):
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=None,
                maturity=dt.datetime(2026, 12, 31),
                currency="EUR",
            )

    def test_option_spec_invalid_option_type(self):
        """Test that invalid option_type raises TypeError."""
        with pytest.raises(ConfigurationError, match="option_type must be OptionType enum"):
            VanillaSpec(
                option_type="CALL",  # Invalid: string instead of enum
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )

    def test_option_spec_invalid_exercise_type(self):
        """Test that invalid exercise_type raises TypeError."""
        with pytest.raises(ConfigurationError, match="exercise_type must be ExerciseType enum"):
            VanillaSpec(
                option_type=OptionType.PUT,
                exercise_type="EUROPEAN",  # Invalid: string instead of enum
                strike=100.0,
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )

    def test_option_spec_frozen(self):
        """Test that VanillaSpec is frozen (immutable)."""
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
        )
        with pytest.raises(AttributeError):
            spec.strike = 105.0


class TestUnderlyingData:
    """Tests for UnderlyingData class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up market environment for tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.maturity, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

    def test_underlying_data_creation(self):
        """Test successful creation of UnderlyingData."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )
        assert ud.initial_value == 100.0
        assert ud.volatility == 0.2
        assert ud.pricing_date == self.pricing_date

    def test_underlying_data_is_frozen(self):
        """Test that UnderlyingData is frozen (immutable)."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )
        with pytest.raises(AttributeError):
            ud.initial_value = 105.0


class TestOptionValuation:
    """Tests for OptionValuation dispatcher class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up market environment and valuation specifications for tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.curve = flat_curve(self.pricing_date, self.maturity, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

        # Standard option spec
        self.call_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        self.put_spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

    def test_option_valuation_with_underlying_data_bsm(self):
        """Test OptionValuation creation with UnderlyingData and BSM pricing."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BSM,
        )
        assert valuation.strike == self.strike
        assert isinstance(valuation._impl, _BSMEuropeanValuation)

    def test_option_valuation_with_underlying_data_binomial(self):
        """Test OptionValuation creation with UnderlyingData and Binomial pricing."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )
        assert isinstance(valuation._impl, _BinomialEuropeanValuation)

    def test_bsm_accepts_nonflat_discount_curve(self):
        """BSM should accept time-varying discount curves."""
        nonflat_curve = DiscountCurve(
            times=np.array([0.0, 0.5, 1.0]),
            dfs=np.array([1.0, np.exp(-0.03 * 0.5), np.exp(-0.06 * 1.0)]),
        )
        market_data = MarketData(self.pricing_date, nonflat_curve, currency="USD")
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BSM,
        )
        assert valuation.present_value() > 0.0

    def test_option_valuation_with_path_simulation_mcs(self):
        """Test OptionValuation creation with PathSimulation and MC pricing."""
        gbm_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(
            paths=1000,
            num_steps=365,
            end_date=self.maturity,
        )
        gbm = GBMProcess(
            self.market_data,
            gbm_params,
            sim_config,
        )

        valuation = OptionValuation(
            underlying=gbm,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )
        assert isinstance(valuation._impl, _MCEuropeanValuation)

    def test_mcs_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price under MC."""
        ex_date = self.pricing_date + dt.timedelta(days=180)
        time_grid = np.array([self.pricing_date, ex_date, self.maturity])

        gbm_params_no_div = GBMParams(initial_value=100.0, volatility=0.2)
        gbm_params_div = GBMParams(
            initial_value=100.0,
            volatility=0.2,
            discrete_dividends=[(ex_date, 1.0)],
        )

        sim_config = SimulationConfig(
            paths=20000,
            time_grid=time_grid,
        )

        gbm_no_div = GBMProcess(
            self.market_data,
            gbm_params_no_div,
            sim_config,
        )
        gbm_div = GBMProcess(
            self.market_data,
            gbm_params_div,
            sim_config,
        )

        val_no_div = OptionValuation(
            underlying=gbm_no_div,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )
        val_div = OptionValuation(
            underlying=gbm_div,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )

        pv_no_div = val_no_div.present_value()
        pv_div = val_div.present_value()

        assert pv_div < pv_no_div

    def test_option_valuation_invalid_maturity(self):
        """Test that OptionValuation raises error if maturity <= pricing_date."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        invalid_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.pricing_date,  # Invalid: maturity == pricing_date
            currency="USD",
        )

        with pytest.raises(ValidationError, match="Option maturity must be after pricing_date"):
            OptionValuation(
                underlying=ud,
                spec=invalid_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_option_valuation_currency_mismatch(self):
        """Test that OptionValuation raises for cross-currency inputs."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        eur_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="EUR",
        )

        with pytest.raises(
            UnsupportedFeatureError, match="Cross-currency valuation is not supported"
        ):
            OptionValuation(
                underlying=ud,
                spec=eur_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_custom_payoff_single_contract_american_supported(self):
        """Custom payoff should be priceable as a single American contract.

        This specifically checks that early exercise decisions compare intrinsic vs continuation
        for the *whole payoff*, not decomposed legs.
        """

        def capped_payoff(spot: np.ndarray | float) -> np.ndarray:
            s = np.asarray(spot, dtype=float)
            return np.minimum(40.0, np.maximum(90.0 - s, 0.0) + np.maximum(s - 110.0, 0.0))

        spec_am = PayoffSpec(
            exercise_type=ExerciseType.AMERICAN,
            maturity=self.maturity,
            payoff_fn=capped_payoff,
            currency="USD",
        )
        spec_eu = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=self.maturity,
            payoff_fn=capped_payoff,
            currency="USD",
        )

        # Binomial (UnderlyingData)
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        pv_binom_am = OptionValuation(
            underlying=ud,
            spec=spec_am,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        ).present_value()
        pv_binom_eu = OptionValuation(
            underlying=ud,
            spec=spec_eu,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        ).present_value()

        assert pv_binom_am >= pv_binom_eu
        assert 0.0 <= pv_binom_am <= 40.0 + 1e-8

        # Monte Carlo (PathSimulation) American via LSM
        gbm_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(paths=20000, num_steps=52, end_date=self.maturity)
        gbm = GBMProcess(self.market_data, gbm_params, sim_config)
        pv_mcs_am = OptionValuation(
            underlying=gbm,
            spec=spec_am,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=3),
        ).present_value()
        pv_mcs_eu = OptionValuation(
            underlying=gbm,
            spec=spec_eu,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        assert pv_mcs_am >= pv_mcs_eu
        assert 0.0 <= pv_mcs_am <= 40.0 + 1e-8

    def test_option_valuation_invalid_pricing_method_type(self):
        """Test that OptionValuation validates pricing_method is PricingMethod enum."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        with pytest.raises(ConfigurationError, match="pricing_method must be PricingMethod enum"):
            OptionValuation(
                underlying=ud,
                spec=self.call_spec,
                pricing_method="BSM",  # Invalid: string instead of enum
            )

    def test_option_valuation_mc_requires_path_simulation(self):
        """Test that Monte Carlo pricing requires PathSimulation, not UnderlyingData."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        with pytest.raises(
            ConfigurationError,
            match="Monte Carlo pricing requires underlying to be a PathSimulation",
        ):
            OptionValuation(
                underlying=ud,
                spec=self.call_spec,
                pricing_method=PricingMethod.MONTE_CARLO,
            )

    def test_underlying_pricing_data_warns_mixed_dividends(self):
        """Both dividend_curve and discrete_dividends should emit a warning."""
        with pytest.warns(UserWarning, match="both dividend_curve and discrete_dividends"):
            UnderlyingData(
                initial_value=100.0,
                volatility=0.2,
                market_data=self.market_data,
                dividend_curve=flat_curve(self.pricing_date, self.maturity, 0.02),
                discrete_dividends=[(self.pricing_date + dt.timedelta(days=90), 1.0)],
            )

    def test_option_valuation_binomial_rejects_path_simulation(self):
        """Test that Binomial pricing rejects PathSimulation (should use UnderlyingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim_config=SimulationConfig(
                paths=1000,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match=r"BINOMIAL pricing requires an UnderlyingData instance.*"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.BINOMIAL,
            )

    def test_option_valuation_bsm_rejects_path_simulation(self):
        """Test that BSM pricing rejects PathSimulation (should use UnderlyingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim_config=SimulationConfig(
                paths=1000,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match=r"BSM pricing requires an UnderlyingData instance.*"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_option_valuation_pde_rejects_path_simulation(self):
        """Test that PDE_FD pricing rejects PathSimulation (should use UnderlyingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim_config=SimulationConfig(
                paths=1000,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match=r"PDE_FD pricing requires an UnderlyingData instance.*"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.PDE_FD,
            )

    def test_option_valuation_american_bsm_not_implemented(self):
        """Test that American option BSM pricing raises NotImplementedError."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        american_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # Error is raised during initialization, not present_value
        with pytest.raises(UnsupportedFeatureError, match="BSM is only applicable to European"):
            OptionValuation(
                underlying=ud,
                spec=american_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_dispatcher_creates_fd_impl_for_american(self):
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(ud, spec, PricingMethod.PDE_FD)
        assert isinstance(valuation._impl, _FDAmericanValuation)

    def test_american_put_fd_close_to_binomial(self):
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=200, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        tree_val = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1200)
        )
        tree_pv = tree_val.present_value()

        # Both are numerical approximations; keep tolerance modest for test stability.
        assert np.isclose(fd_pv, tree_pv, rtol=0.01)

    def test_american_call_fd_close_to_binomial(self):
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=flat_curve(self.pricing_date, self.maturity, 0.03),
        )
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=200, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        tree_val = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1200)
        )
        tree_pv = tree_val.present_value()

        # Both are numerical approximations; keep tolerance modest for test stability.
        assert np.isclose(fd_pv, tree_pv, rtol=0.01)

    def test_american_call_no_dividend_equals_european_fd(self):
        """American CALL with q=0 has no early exercise premium.

        Regression test for the PDE fast-path that prices this case using the
        European CN solver (skipping PSOR).
        """
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec_am = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        spec_eu = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        params = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)

        am_val = OptionValuation(ud, spec_am, PricingMethod.PDE_FD, params=params)
        eu_val = OptionValuation(ud, spec_eu, PricingMethod.PDE_FD, params=params)

        am_pv = am_val.present_value()
        eu_pv = eu_val.present_value()

        assert np.isclose(am_pv, eu_pv, rtol=1e-12, atol=1e-12)

    def test_american_call_no_dividend_close_to_bsm_european(self):
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec_am = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        spec_eu = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec_am,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=200, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        bsm_val = OptionValuation(ud, spec_eu, PricingMethod.BSM)
        bsm_pv = bsm_val.present_value()

        # American call with no dividends should be near European (no early exercise benefit).
        assert np.isclose(fd_pv, bsm_pv, atol=1.0)

    def test_binomial_discrete_dividends_close_to_bsm(self):
        """Binomial with discrete dividends should be close to BSM with dividend-adjusted spot."""
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        discrete_divs = [
            (self.pricing_date + dt.timedelta(days=90), 0.6),
            (self.pricing_date + dt.timedelta(days=180), 0.6),
        ]

        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
            discrete_dividends=discrete_divs,
        )

        binom_val = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=1200),
        )
        bsm_val = OptionValuation(ud, spec, PricingMethod.BSM)

        binom_pv = binom_val.present_value()
        bsm_pv = bsm_val.present_value()

        assert np.isclose(binom_pv, bsm_pv, rtol=0.02)

    def test_intraday_timestamps_price_european_across_engines(self):
        """Intraday datetimes should price consistently across European engines."""
        pricing_date = dt.datetime(2025, 1, 1, 10, 30, 15)
        maturity = dt.datetime(2026, 1, 1, 15, 45, 5)

        md = MarketData(pricing_date, flat_curve(pricing_date, maturity, 0.05), currency="USD")
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=md,
            dividend_curve=flat_curve(pricing_date, maturity, 0.02),
        )
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=maturity,
            currency="USD",
        )

        bsm = OptionValuation(ud, spec, PricingMethod.BSM).present_value()
        binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        ).present_value()
        pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=220, time_steps=360),
        ).present_value()

        gbm = GBMProcess(
            md,
            GBMParams(
                initial_value=100.0,
                volatility=0.2,
                dividend_curve=flat_curve(pricing_date, maturity, 0.02),
            ),
            SimulationConfig(paths=50_000, num_steps=252, end_date=maturity),
        )
        mc = OptionValuation(
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=7),
        ).present_value()

        for value in (bsm, binom, pde, mc):
            assert np.isfinite(value)
            assert value > 0.0

        assert np.isclose(bsm, binom, rtol=0.01)
        assert np.isclose(bsm, pde, rtol=0.01)
        assert np.isclose(bsm, mc, atol=0.2)

    def test_intraday_timestamps_price_american_across_engines(self):
        """Intraday datetimes should price consistently across American engines."""
        pricing_date = dt.datetime(2025, 1, 3, 9, 47, 12)
        maturity = dt.datetime(2025, 10, 3, 14, 21, 33)

        md = MarketData(pricing_date, flat_curve(pricing_date, maturity, 0.04), currency="USD")
        ud = UnderlyingData(
            initial_value=95.0,
            volatility=0.25,
            market_data=md,
            dividend_curve=flat_curve(pricing_date, maturity, 0.01),
        )
        spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=100.0,
            maturity=maturity,
            currency="USD",
        )

        binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=700),
        ).present_value()
        pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=260, time_steps=420),
        ).present_value()

        gbm = GBMProcess(
            md,
            GBMParams(
                initial_value=95.0,
                volatility=0.25,
                dividend_curve=flat_curve(pricing_date, maturity, 0.01),
            ),
            SimulationConfig(paths=50_000, num_steps=252, end_date=maturity),
        )
        mc = OptionValuation(
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=11, deg=3),
        ).present_value()

        for value in (binom, pde, mc):
            assert np.isfinite(value)
            assert value > 0.0

        assert np.isclose(binom, pde, rtol=0.01)
        assert np.isclose(binom, mc, atol=0.2)

    def test_market_data_rejects_timezone_aware_pricing_date(self):
        """Timezone-aware pricing_date should be rejected."""
        aware_pricing_date = dt.datetime(2025, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

        with pytest.raises(ValidationError, match="pricing_date must be timezone-naive datetime"):
            MarketData(
                pricing_date=aware_pricing_date,
                discount_curve=self.curve,
                currency="USD",
            )

    def test_vanilla_spec_rejects_timezone_aware_maturity(self):
        """Timezone-aware spec maturities should be rejected."""
        aware_maturity = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

        with pytest.raises(ValidationError, match="maturity must be timezone-naive datetime"):
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=aware_maturity,
                currency="USD",
            )

    def test_asian_spec_rejects_timezone_aware_fixing_dates(self):
        """Timezone-aware fixing dates should be rejected for Asian specs."""
        maturity = dt.datetime(2026, 1, 1, 12, 0, 0)
        aware_fixing = dt.datetime(2025, 6, 1, 9, 0, tzinfo=dt.timezone.utc)

        with pytest.raises(ValidationError, match="fixing_dates entry must be timezone-naive"):
            AsianSpec(
                averaging=AsianAveraging.GEOMETRIC,
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=maturity,
                fixing_dates=(aware_fixing,),
            )

    def test_calculate_year_fraction_rejects_mixed_naive_and_aware(self):
        """Mixed naive/aware timestamps should be rejected with ValidationError."""
        start_date = dt.datetime(2025, 1, 1, 9, 30, 0)
        end_date = dt.datetime(2025, 6, 1, 9, 30, 0, tzinfo=dt.timezone.utc)

        with pytest.raises(ValidationError, match="end_date must be timezone-naive datetime"):
            calculate_year_fraction(start_date, end_date)

    def test_simulation_config_rejects_timezone_aware_end_date(self):
        """Timezone-aware end_date should be rejected in SimulationConfig."""
        aware_end = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
        with pytest.raises(ValidationError, match="end_date must be timezone-naive datetime"):
            SimulationConfig(paths=1000, num_steps=12, end_date=aware_end)

    def test_simulation_config_rejects_timezone_aware_time_grid_entry(self):
        """Timezone-aware entries in explicit time_grid should be rejected."""
        time_grid = np.array(
            [
                dt.datetime(2025, 1, 1, 9, 30),
                dt.datetime(2025, 6, 1, 9, 30, tzinfo=dt.timezone.utc),
            ],
            dtype=object,
        )
        with pytest.raises(ValidationError, match="time_grid entry must be timezone-naive"):
            SimulationConfig(paths=1000, time_grid=time_grid)


class TestCrossMethodGridAndParamCombos:
    """Cross-method comparison grid and interaction-style parameter coverage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.currency = "USD"
        self.spot = 100.0
        self.strike = 100.0
        self.vol = 0.2

    def _curve(self, kind: str) -> DiscountCurve | None:
        if kind == "none":
            return None
        if kind == "flat":
            return flat_curve(self.pricing_date, self.maturity, 0.04)
        if kind == "nonflat":
            return DiscountCurve.from_forwards(
                times=np.array([0.0, 0.5, 1.0]),
                forwards=np.array([0.02, 0.06]),
            )
        raise ValueError(f"unknown curve kind: {kind}")

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("r_curve_kind", ["flat", "nonflat"])
    @pytest.mark.parametrize("div_curve_kind", ["none", "flat", "nonflat"])
    def test_european_cross_method_grid(self, option_type, r_curve_kind, div_curve_kind):
        r_curve = self._curve(r_curve_kind)
        assert r_curve is not None
        div_curve = self._curve(div_curve_kind)

        md = MarketData(self.pricing_date, r_curve, currency=self.currency)
        ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.vol,
            market_data=md,
            dividend_curve=div_curve,
        )
        spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
        )

        pv_bsm = OptionValuation(ud, spec, PricingMethod.BSM).present_value()
        pv_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=900),
        ).present_value()
        pv_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=200, max_iter=20_000),
        ).present_value()

        assert np.isfinite(pv_bsm) and pv_bsm > 0.0
        assert np.isfinite(pv_binom) and pv_binom > 0.0
        assert np.isfinite(pv_pde) and pv_pde > 0.0
        assert np.isclose(pv_binom, pv_bsm, rtol=0.03)
        assert np.isclose(pv_pde, pv_bsm, rtol=0.03)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("div_curve_kind", ["none", "flat"])
    def test_american_cross_method_grid(self, option_type, div_curve_kind):
        r_curve = flat_curve(self.pricing_date, self.maturity, 0.04)
        md = MarketData(self.pricing_date, r_curve, currency=self.currency)
        ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.vol,
            market_data=md,
            dividend_curve=self._curve(div_curve_kind),
        )
        spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
        )

        pv_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=1200),
        ).present_value()
        pv_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=200, max_iter=20_000),
        ).present_value()

        assert np.isfinite(pv_binom) and pv_binom > 0.0
        assert np.isfinite(pv_pde) and pv_pde > 0.0
        assert np.isclose(pv_pde, pv_binom, rtol=0.03)

    @pytest.mark.parametrize(
        "mc_params",
        [
            MonteCarloParams(random_seed=42, deg=2, ridge_lambda=0.0, min_itm=10),
            MonteCarloParams(random_seed=42, deg=3, ridge_lambda=1e-8, min_itm=25),
            MonteCarloParams(random_seed=42, deg=5, ridge_lambda=1e-6, min_itm=50),
        ],
    )
    def test_monte_carlo_parameter_interactions(self, mc_params: MonteCarloParams):
        """Regression degrees and regularization combos should produce finite PVs."""
        md = MarketData(
            self.pricing_date,
            flat_curve(self.pricing_date, self.maturity, 0.04),
            currency=self.currency,
        )
        gbm = GBMProcess(
            md,
            GBMParams(initial_value=self.spot, volatility=self.vol),
            SimulationConfig(paths=30_000, num_steps=52, end_date=self.maturity),
        )
        spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
        )
        pv = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=mc_params).present_value()
        assert np.isfinite(pv)
        assert pv > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# DCC monotonicity: ACT/360 yields a longer year fraction than ACT/365F for
# the same calendar period, so option prices must be strictly higher.
# ═══════════════════════════════════════════════════════════════════════════


class TestDayCountConventionMonotonicity:
    """ACT/360 yields a longer year fraction than ACT/365F for the same calendar
    period, so option values must be strictly higher under ACT/360."""

    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    vol = 0.20
    rate = 0.05
    div_yield = 0.02
    currency = "USD"

    _LO = DayCountConvention.ACT_365F
    _HI = DayCountConvention.ACT_360

    def _curve(self, rate: float, dcc: DayCountConvention) -> DiscountCurve:
        ttm = calculate_year_fraction(self.pricing_date, self.maturity, dcc)
        return DiscountCurve.flat(rate, end_time=ttm)

    def _md(self, dcc: DayCountConvention) -> MarketData:
        return MarketData(
            self.pricing_date,
            self._curve(self.rate, dcc),
            currency=self.currency,
            day_count_convention=dcc,
        )

    def _ud(self, dcc: DayCountConvention) -> UnderlyingData:
        return UnderlyingData(
            initial_value=self.spot,
            volatility=self.vol,
            market_data=self._md(dcc),
            dividend_curve=self._curve(self.div_yield, dcc),
        )

    def _gbm(self, dcc: DayCountConvention) -> GBMProcess:
        return GBMProcess(
            self._md(dcc),
            GBMParams(
                initial_value=self.spot,
                volatility=self.vol,
                dividend_curve=self._curve(self.div_yield, dcc),
            ),
            SimulationConfig(paths=200_000, end_date=self.maturity, num_steps=200),
        )

    def _pv(
        self,
        dcc: DayCountConvention,
        option_spec: VanillaSpec,
        method: PricingMethod,
        params: BinomialParams | MonteCarloParams | PDEParams | None = None,
    ) -> float:
        if method is PricingMethod.MONTE_CARLO:
            ud = self._gbm(dcc)
        else:
            ud = self._ud(dcc)
        return OptionValuation(ud, option_spec, method, params=params).present_value()

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize(
        "exercise_type,method,params",
        [
            (ExerciseType.EUROPEAN, PricingMethod.BSM, None),
            (
                ExerciseType.EUROPEAN,
                PricingMethod.PDE_FD,
                PDEParams(spot_steps=140, time_steps=140, max_iter=20_000),
            ),
            (ExerciseType.EUROPEAN, PricingMethod.BINOMIAL, BinomialParams(num_steps=500)),
            (ExerciseType.EUROPEAN, PricingMethod.MONTE_CARLO, MonteCarloParams(random_seed=42)),
            (
                ExerciseType.AMERICAN,
                PricingMethod.PDE_FD,
                PDEParams(spot_steps=140, time_steps=140, max_iter=20_000),
            ),
            (ExerciseType.AMERICAN, PricingMethod.BINOMIAL, BinomialParams(num_steps=500)),
            (ExerciseType.AMERICAN, PricingMethod.MONTE_CARLO, MonteCarloParams(random_seed=42)),
        ],
        ids=["bsm_eu", "pde_eu", "binom_eu", "mc_eu", "pde_am", "binom_am", "mc_am"],
    )
    def test_act360_exceeds_act365f(self, option_type, exercise_type, method, params):
        """ACT/360 (T ≈ 1.014) yields a strictly higher PV than ACT/365F (T = 1.0)."""
        option_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=exercise_type,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
        )
        pv_lo = self._pv(self._LO, option_spec, method, params)
        pv_hi = self._pv(self._HI, option_spec, method, params)
        assert pv_hi > pv_lo, (
            f"ACT/360 {pv_hi:.6f} should exceed ACT/365F {pv_lo:.6f} "
            f"({method.name} {exercise_type.value} {option_type.value})"
        )


class TestAsUnderlyingData:
    """Tests for the as_underlying_data conversion function."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.md = MarketData(
            dt.datetime(2025, 1, 1),
            DiscountCurve.flat(0.05, end_time=1.0),
            currency="USD",
        )
        self.sim_cfg = SimulationConfig(paths=1_000, end_date=dt.datetime(2026, 1, 1), num_steps=10)

    def test_gbm_attributes_preserved(self) -> None:
        """Converted UnderlyingData retains all GBM attributes."""
        div_curve = DiscountCurve.flat(0.02, end_time=1.0)
        divs = [(dt.datetime(2025, 6, 1), 1.50)]
        gbm = GBMProcess(
            self.md,
            GBMParams(
                initial_value=100.0,
                volatility=0.25,
                dividend_curve=div_curve,
                discrete_dividends=divs,
            ),
            self.sim_cfg,
        )
        ud = as_underlying_data(gbm)

        assert isinstance(ud, UnderlyingData)
        assert ud.initial_value == gbm.initial_value
        assert ud.volatility == gbm.volatility
        assert ud.market_data is gbm.market_data
        assert ud.dividend_curve is gbm.dividend_curve
        assert ud.discrete_dividends == tuple(divs)

    def test_underlying_data_passthrough(self) -> None:
        """UnderlyingData input is returned unchanged."""
        ud = UnderlyingData(initial_value=100.0, volatility=0.25, market_data=self.md)
        result = as_underlying_data(ud)
        assert result is ud

    def test_rejects_jd_process(self) -> None:
        """JDProcess must be rejected — jump parameters would be silently lost."""
        jd = JDProcess(
            self.md,
            JDParams(initial_value=100.0, volatility=0.25, lambd=0.5, mu=-0.1, delta=0.2),
            self.sim_cfg,
        )
        with pytest.raises(ConfigurationError, match="GBMProcess"):
            as_underlying_data(jd)

    def test_rejects_srd_process(self) -> None:
        """SRDProcess must be rejected — mean-reversion parameters would be silently lost."""
        srd = SRDProcess(
            self.md,
            SRDParams(initial_value=0.04, volatility=0.1, kappa=2.0, theta=0.04),
            self.sim_cfg,
        )
        with pytest.raises(ConfigurationError, match="GBMProcess"):
            as_underlying_data(srd)
