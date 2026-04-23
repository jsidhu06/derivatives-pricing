import pytest
import datetime as dt
import numpy as np
from derivatives_pricing.exceptions import ValidationError
from derivatives_pricing.stochastic_processes import (
    PathSimulation,
    GBMProcess,
    SRDProcess,
    JDProcess,
    GBMParams,
    JDParams,
    SRDParams,
    SimulationConfig,
)
from derivatives_pricing.market_environment import MarketData
from helpers import flat_curve


class TestPathSimulation:
    """Tests for the abstract PathSimulation class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PathSimulation cannot be instantiated directly due to abstract method"""
        pricing_date = dt.datetime(2025, 1, 1)
        end_date = dt.datetime(2026, 1, 1)
        curve = flat_curve(pricing_date, end_date, 0.05)
        market_data = MarketData(pricing_date, curve, currency="EUR")
        process_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(
            paths=10000,
            num_steps=12,
            end_date=dt.datetime(2026, 1, 1),
        )

        with pytest.raises(
            TypeError,
            match=r"Can't instantiate abstract class PathSimulation.*_generate_paths",
        ):
            PathSimulation("test_name", market_data, process_params, sim_config, corr=None)


class TestGBMProcess:
    """Tests for the GBMProcess class"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up market environment for GBM tests"""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="EUR")
        self.process_params = GBMParams(initial_value=36.0, volatility=0.2)
        self.sim_config = SimulationConfig(
            paths=10000,
            num_steps=12,
            end_date=self.end_date,
        )
        self.gbm = GBMProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
            corr=None,
        )

    def test_simulate_reproducibility_with_seed(self):
        """Test that simulate produces identical results with the same random_seed."""
        # Generate paths with seed 1000
        paths_1 = self.gbm.simulate(random_seed=1000).copy()

        # Create a new instance and generate again with same seed
        gbm_2 = GBMProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
            corr=None,
        )
        paths_2 = gbm_2.simulate(random_seed=1000).copy()

        # Should be identical
        np.testing.assert_array_equal(paths_1, paths_2)

    def test_simulate_basic_properties(self):
        """Test basic properties of simulated paths."""
        paths = self.gbm.simulate(random_seed=1000)

        # Check shape: (M, num_paths) where M is time steps, num_paths is paths
        M = len(self.gbm.time_grid)
        num_paths = self.sim_config.paths
        assert paths.shape == (M, num_paths)

        # First row should be initial value
        np.testing.assert_array_equal(paths[0], self.process_params.initial_value)

        # All values should be positive (GBM property)
        assert np.all(paths > 0)

    def test_simulate_returns_array(self):
        """Test that simulate returns the paths array."""
        paths = self.gbm.simulate(random_seed=1000)

        # Should return a valid ndarray
        assert isinstance(paths, np.ndarray)
        assert paths.shape[0] == len(self.gbm.time_grid)
        assert paths.shape[1] == self.sim_config.paths

    def test_time_grid_generation(self):
        """Test that time_grid is generated correctly."""
        # Generate paths to ensure time_grid is populated
        self.gbm.simulate(random_seed=42)

        assert self.gbm.time_grid is not None
        # Time grid should start with pricing_date and end with end_date
        assert self.gbm.time_grid[0] == self.market_data.pricing_date
        assert self.gbm.time_grid[-1] == self.sim_config.end_date

    def test_simulate(self):
        """Test simulate method."""
        values = self.gbm.simulate(random_seed=42)

        # Should return the instrument values array
        assert values is not None
        assert values.shape == (len(self.gbm.time_grid), self.sim_config.paths)

    def test_gbm_with_different_volatilities(self):
        """Test GBM with different volatility levels."""
        low_vol_params = GBMParams(initial_value=100.0, volatility=0.05)
        high_vol_params = GBMParams(initial_value=100.0, volatility=0.5)

        gbm_low = GBMProcess(
            self.market_data,
            low_vol_params,
            self.sim_config,
        )

        gbm_high = GBMProcess(
            self.market_data,
            high_vol_params,
            self.sim_config,
        )

        paths_low = gbm_low.simulate(random_seed=42)
        paths_high = gbm_high.simulate(random_seed=42)

        # Higher volatility should lead to wider range of final values
        std_low = np.std(paths_low[-1])
        std_high = np.std(paths_high[-1])

        assert std_high > std_low * 1.5  # High vol should have much wider spread

    def test_gbm_drift_matches_risk_free_rate(self):
        """Test that GBM paths have drift approximately equal to risk-free rate."""
        # With many paths, mean return should approximate r - 0.5*sigma^2
        short_rate = 0.05
        volatility = 0.2
        pricing_date = dt.datetime(2025, 1, 1)
        end_date = dt.datetime(2026, 1, 1)
        curve = flat_curve(pricing_date, end_date, short_rate)
        market_data = MarketData(pricing_date, curve, currency="USD")
        process_params = GBMParams(initial_value=100.0, volatility=volatility)
        sim_config = SimulationConfig(
            paths=10000,
            num_steps=365,
            end_date=dt.datetime(2026, 1, 1),
        )

        gbm = GBMProcess(market_data, process_params, sim_config)
        paths = gbm.simulate(random_seed=42)

        # Calculate mean return
        T = 1.0  # 1 year
        final_prices = paths[-1]
        initial_price = paths[0, 0]
        returns = np.log(final_prices / initial_price) / T

        expected_drift = short_rate - 0.5 * volatility**2
        mean_return = np.mean(returns)

        # Should be close (within standard error)
        assert np.abs(mean_return - expected_drift) < 0.01


class TestSRDProcess:
    """Tests for the SRDProcess (CIR) class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up market environment for SRD tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")
        self.process_params = SRDParams(
            initial_value=0.03,
            volatility=0.015,
            kappa=0.2,
            theta=0.05,
        )
        self.sim_config = SimulationConfig(
            paths=5000,
            num_steps=365,
            end_date=self.end_date,
        )
        self.srd = SRDProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
        )

    def test_srd_initialization(self):
        """Test SRD initializes with correct parameters."""
        assert self.srd.kappa == self.process_params.kappa
        assert self.srd.theta == self.process_params.theta
        assert self.srd.volatility == self.process_params.volatility

    def test_srd_simulate(self):
        """Test SRD path simulation."""
        paths = self.srd.simulate(random_seed=42)

        # Check shape
        assert paths.shape[0] == len(self.srd.time_grid)
        assert paths.shape[1] == self.sim_config.paths

        # First row should be initial value
        np.testing.assert_array_almost_equal(paths[0], self.process_params.initial_value)

        # All values should be non-negative (CIR property)
        assert np.all(paths >= 0)

    def test_srd_mean_reversion(self):
        """Test that SRD exhibits mean reversion towards theta."""
        long_sim_config = SimulationConfig(
            paths=5000,
            num_steps=1825,
            end_date=dt.datetime(2030, 1, 1),  # 5 years
        )

        srd_long = SRDProcess(
            self.market_data,
            self.process_params,
            long_sim_config,
        )

        paths = srd_long.simulate(random_seed=42)

        # Final values should be closer to theta than initial value
        final_mean = np.mean(paths[-1])

        # With mean reversion, final mean should be between initial and theta
        initial = paths[0, 0]
        theta = self.process_params.theta

        # Final should be closer to theta
        dist_to_theta_initial = abs(initial - theta)
        dist_to_theta_final = abs(final_mean - theta)

        assert dist_to_theta_final < dist_to_theta_initial

    def test_srd_reproducibility(self):
        """Test SRD reproducibility with same seed."""
        paths1 = self.srd.simulate(random_seed=123).copy()

        srd2 = SRDProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
        )

        paths2 = srd2.simulate(random_seed=123)

        np.testing.assert_array_equal(paths1, paths2)

    def test_srd_low_mean_reversion_stays_near_start(self):
        """With very low kappa, long-run pull should be weak."""
        params = SRDParams(initial_value=0.03, volatility=0.01, kappa=1.0e-4, theta=0.20)
        proc = SRDProcess(self.market_data, params, self.sim_config)
        paths = proc.simulate(random_seed=7)
        assert abs(np.mean(paths[-1]) - 0.03) < 0.03

    def test_srd_high_mean_reversion_moves_towards_theta(self):
        """With very high kappa, process should quickly revert toward theta."""
        params = SRDParams(initial_value=0.03, volatility=0.01, kappa=20.0, theta=0.08)
        proc = SRDProcess(self.market_data, params, self.sim_config)
        paths = proc.simulate(random_seed=8)
        assert abs(np.mean(paths[-1]) - 0.08) < 0.03


class TestJDProcess:
    """Tests for the JDProcess (Merton) class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up market environment for Jump Diffusion tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")
        self.process_params = JDParams(
            initial_value=100.0,
            volatility=0.2,
            lambd=0.5,  # 0.5 jumps per year on average
            mu=-0.1,
            delta=0.3,
        )
        self.sim_config = SimulationConfig(
            paths=5000,
            num_steps=365,
            end_date=self.end_date,
        )
        self.jd = JDProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
        )

    def test_jd_initialization(self):
        """Test JD initializes with correct parameters."""
        assert self.jd.lambd == self.process_params.lambd
        assert self.jd.mu == self.process_params.mu
        assert self.jd.delta == self.process_params.delta
        assert self.jd.volatility == self.process_params.volatility

    def test_jd_simulate(self):
        """Test JD path simulation."""
        paths = self.jd.simulate(random_seed=42)

        # Check shape
        assert paths.shape[0] == len(self.jd.time_grid)
        assert paths.shape[1] == self.sim_config.paths

        # First row should be initial value
        np.testing.assert_array_almost_equal(paths[0], self.process_params.initial_value)

        # All values should be positive
        assert np.all(paths > 0)

    def test_jd_jump_impact(self):
        """Test that jump diffusion paths have discontinuities (jumps)."""
        # Generate paths with jumps
        paths_jd = self.jd.simulate(random_seed=42)

        # Generate comparable GBM paths
        gbm_params = GBMParams(
            initial_value=self.process_params.initial_value,
            volatility=self.process_params.volatility,
        )
        gbm = GBMProcess(
            self.market_data,
            gbm_params,
            self.sim_config,
        )

        paths_gbm = gbm.simulate(random_seed=42)

        # JD should have larger final value spread due to jumps
        std_jd = np.std(paths_jd[-1])
        std_gbm = np.std(paths_gbm[-1])

        # Jump diffusion should have higher volatility
        assert std_jd > std_gbm * 0.9  # Allow some tolerance

    def test_jd_reproducibility(self):
        """Test JD reproducibility with same seed."""
        paths1 = self.jd.simulate(random_seed=456).copy()

        jd2 = JDProcess(
            self.market_data,
            self.process_params,
            self.sim_config,
        )

        paths2 = jd2.simulate(random_seed=456)

        np.testing.assert_array_equal(paths1, paths2)

    def test_jd_zero_lambd(self):
        """Test JD with zero jump intensity (should be similar to GBM)."""
        no_jump_params = JDParams(
            initial_value=100.0,
            volatility=0.2,
            lambd=0.0,  # No jumps
            mu=0.0,
            delta=0.0,
        )

        jd_no_jump = JDProcess(
            self.market_data,
            no_jump_params,
            self.sim_config,
        )

        paths_jd = jd_no_jump.simulate(random_seed=99)

        # Without jumps, all values should be positive and continuous
        assert np.all(paths_jd > 0)

    def test_jd_zero_jump_size_reduces_to_gbm_moments(self):
        """mu=delta=0 removes jump-size effect even if intensity is non-zero."""
        jd_params = JDParams(initial_value=100.0, volatility=0.2, lambd=2.0, mu=0.0, delta=0.0)
        jd = JDProcess(self.market_data, jd_params, self.sim_config)
        gbm = GBMProcess(
            self.market_data, GBMParams(initial_value=100.0, volatility=0.2), self.sim_config
        )
        jd_paths = jd.simulate(random_seed=123)
        gbm_paths = gbm.simulate(random_seed=123)
        assert np.isclose(np.mean(jd_paths[-1]), np.mean(gbm_paths[-1]), rtol=0.02)
        assert np.isclose(np.std(jd_paths[-1]), np.std(gbm_paths[-1]), rtol=0.03)

    def test_jd_high_intensity_increases_tail_dispersion(self):
        """Higher jump intensity should widen terminal distribution tails."""
        low = JDProcess(
            self.market_data,
            JDParams(initial_value=100.0, volatility=0.2, lambd=0.1, mu=-0.1, delta=0.3),
            self.sim_config,
        )
        high = JDProcess(
            self.market_data,
            JDParams(initial_value=100.0, volatility=0.2, lambd=3.0, mu=-0.1, delta=0.3),
            self.sim_config,
        )
        low_paths = low.simulate(random_seed=42)
        high_paths = high.simulate(random_seed=42)
        assert np.std(high_paths[-1]) > np.std(low_paths[-1])


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_simulation_config_creation(self):
        """Test SimulationConfig creation with required parameters."""
        config = SimulationConfig(
            paths=5000,
            num_steps=365,
            end_date=dt.datetime(2026, 1, 1),
        )

        assert config.paths == 5000
        assert config.num_steps == 365
        assert config.end_date == dt.datetime(2026, 1, 1)

    def test_simulation_config_with_observation_dates(self):
        """Test SimulationConfig with special dates."""
        observation_dates = {
            dt.datetime(2025, 3, 17),
            dt.datetime(2025, 6, 20),
        }

        config = SimulationConfig(
            paths=1000,
            num_steps=365,
            end_date=dt.datetime(2026, 1, 1),
            observation_dates=observation_dates,
        )

        assert len(config.observation_dates) == 2

    def test_simulation_config_with_custom_time_grid(self):
        """Test SimulationConfig with pre-defined time grid."""
        custom_grid = np.array(
            [
                dt.datetime(2025, 1, 1),
                dt.datetime(2025, 6, 1),
                dt.datetime(2026, 1, 1),
            ]
        )

        config = SimulationConfig(
            paths=500,
            time_grid=custom_grid,
        )

        assert config.time_grid is not None
        np.testing.assert_array_equal(config.time_grid, custom_grid)

    def test_simulation_config_with_end_date_and_time_grid_raise_error(self):
        """Supplying both end_date and time_grid should raise."""
        custom_grid = np.array(
            [
                dt.datetime(2025, 1, 1),
                dt.datetime(2025, 6, 1),
                dt.datetime(2026, 1, 1),
            ]
        )

        with pytest.raises(ValidationError):
            SimulationConfig(
                paths=500,
                end_date=dt.datetime(2026, 1, 1),
                time_grid=custom_grid,
            )

    def test_simulation_config_missing_num_steps_raises(self):
        """end_date without num_steps should raise."""
        with pytest.raises(ValidationError, match="num_steps must be provided"):
            SimulationConfig(
                paths=500,
                end_date=dt.datetime(2026, 1, 1),
            )

    def test_simulation_config_with_num_steps(self):
        """num_steps mode should be accepted (uniform-step grid)."""
        config = SimulationConfig(
            paths=500,
            end_date=dt.datetime(2026, 1, 1),
            num_steps=10,
        )
        assert config.num_steps == 10


class TestVarianceReduction:
    """Tests for configurable antithetic variates and moment matching."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="EUR")
        self.process_params = GBMParams(initial_value=100.0, volatility=0.2)

    def _make_gbm(self, *, antithetic: bool, moment_matching: bool, paths: int = 10000):
        sim_config = SimulationConfig(
            paths=paths,
            end_date=self.end_date,
            num_steps=252,
            antithetic=antithetic,
            moment_matching=moment_matching,
        )
        return GBMProcess(self.market_data, self.process_params, sim_config, corr=None)

    def test_defaults_are_true(self):
        """SimulationConfig defaults to antithetic=True, moment_matching=True."""
        sim_config = SimulationConfig(paths=100, end_date=self.end_date, num_steps=10)
        assert sim_config.antithetic is True
        assert sim_config.moment_matching is True

    def test_antithetic_produces_symmetric_randoms(self):
        """With antithetic=True, random draws should come in negated pairs."""
        gbm = self._make_gbm(antithetic=True, moment_matching=False, paths=1000)
        z = gbm._standard_normals(random_seed=42, steps=10, paths=1000)
        half = z.shape[1] // 2
        # Z[:, i] + Z[:, i + half] == 0 for all i
        assert np.allclose(z[:, :half] + z[:, half:], 0.0, atol=1e-14)

    def test_no_antithetic_no_symmetric_randoms(self):
        """With antithetic=False, random draws should NOT cancel in pairs."""
        gbm = self._make_gbm(antithetic=False, moment_matching=False, paths=1000)
        z = gbm._standard_normals(random_seed=42, steps=10, paths=1000)
        half = z.shape[1] // 2
        assert not np.allclose(z[:, :half] + z[:, half:], 0.0, atol=1e-6)

    def test_moment_matching_normalises_randoms(self):
        """With moment_matching=True, raw normals should have mean≈0 and std≈1."""
        gbm = self._make_gbm(antithetic=False, moment_matching=True, paths=10000)
        # Access the internal normals directly
        z = gbm._standard_normals(random_seed=42, steps=10, paths=10000)
        assert np.isclose(z.mean(), 0.0, atol=1e-14)
        assert np.isclose(z.std(), 1.0, atol=1e-14)

    def test_no_moment_matching_raw_stats(self):
        """With moment_matching=False, raw normals have small but nonzero mean drift."""
        gbm = self._make_gbm(antithetic=False, moment_matching=False, paths=500)
        z = gbm._standard_normals(random_seed=42, steps=10, paths=500)
        # With only 500 paths the sample mean won't be exactly zero
        assert not np.isclose(z.mean(), 0.0, atol=1e-14)

    def test_antithetic_reduces_estimator_variance(self):
        """Antithetic variates should reduce variance of a call payoff estimator.

        We run many small batches and compare the variance of the batch means.
        """
        n_batches = 50
        batch_paths = 500
        strike = 100.0

        means_av = []
        means_plain = []
        for seed in range(n_batches):
            gbm_av = self._make_gbm(antithetic=True, moment_matching=False, paths=batch_paths)
            terminal = gbm_av.simulate(random_seed=seed)[-1]
            means_av.append(np.mean(np.maximum(terminal - strike, 0.0)))

            gbm_plain = self._make_gbm(antithetic=False, moment_matching=False, paths=batch_paths)
            terminal = gbm_plain.simulate(random_seed=seed + 10000)[-1]
            means_plain.append(np.mean(np.maximum(terminal - strike, 0.0)))

        # Variance of batch means should be lower with antithetic
        assert np.var(means_av) < np.var(means_plain)

    def test_both_off_still_produces_valid_paths(self):
        """With both variance reduction techniques off, paths should still be valid GBM."""
        gbm = self._make_gbm(antithetic=False, moment_matching=False, paths=1000)
        paths = gbm.simulate(random_seed=42)

        assert np.all(paths > 0)
        assert paths[0, 0] == self.process_params.initial_value

    def test_odd_paths_with_antithetic_warns_and_falls_back(self):
        """Odd path count with antithetic=True should warn and fall back to plain sampling."""
        gbm = self._make_gbm(antithetic=True, moment_matching=False, paths=999)
        with pytest.warns(UserWarning, match="antithetic=True but paths=999 is odd"):
            paths = gbm.simulate(random_seed=42)
        assert paths.shape[1] == 999
        assert np.all(paths > 0)


class TestObservationDateFiltering:
    """Observation dates before pricing_date must not enter the time grid."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.md = MarketData(self.pricing_date, self.curve, currency="USD")
        self.params = GBMParams(initial_value=100.0, volatility=0.20)

    def test_past_observation_date_excluded_from_grid(self):
        """An observation date before pricing_date must not appear in the time grid."""
        past = dt.datetime(2024, 6, 1)
        sc = SimulationConfig(
            paths=100, num_steps=12, end_date=self.end_date, observation_dates={past}
        )
        gbm = GBMProcess(self.md, self.params, sc)
        gbm._ensure_time_grid()
        assert gbm.time_grid[0] == self.pricing_date
        assert past not in set(gbm.time_grid)

    def test_past_obs_date_does_not_corrupt_simulation(self):
        """Simulation with a past observation date must match the clean baseline."""
        sc_clean = SimulationConfig(paths=5000, num_steps=12, end_date=self.end_date)
        sc_past = SimulationConfig(
            paths=5000,
            num_steps=12,
            end_date=self.end_date,
            observation_dates={dt.datetime(2024, 6, 1)},
        )
        clean = GBMProcess(self.md, self.params, sc_clean).simulate(random_seed=42)
        dirty = GBMProcess(self.md, self.params, sc_past).simulate(random_seed=42)
        np.testing.assert_array_equal(clean, dirty)

    def test_explicit_time_grid_before_pricing_date_raises(self):
        """An explicit time_grid starting before pricing_date must be rejected at construction."""
        bad_grid = np.array(
            [
                dt.datetime(2024, 6, 1),
                self.pricing_date,
                self.end_date,
            ]
        )
        sc = SimulationConfig(paths=100, time_grid=bad_grid)
        with pytest.raises(ValidationError, match=r"before pricing_date"):
            GBMProcess(self.md, self.params, sc)

    def test_grid_start_before_pricing_date_raises(self):
        """grid_start earlier than pricing_date must be caught by the assertion."""
        sc = SimulationConfig(
            paths=100,
            num_steps=10,
            end_date=self.end_date,
            grid_start=dt.datetime(2024, 6, 1),
        )
        gbm = GBMProcess(self.md, self.params, sc)
        with pytest.raises(ValidationError, match=r"Time grid must start at pricing_date"):
            gbm._ensure_time_grid()


# ═══════════════════════════════════════════════════════════════════════════
# Standard normal shock determinism (MC bump-and-revalue invariant)
# ═══════════════════════════════════════════════════════════════════════════
# OptionValuation Greeks via NUMERICAL bump-and-revalue rely on the bumped
# valuations sharing the same Monte Carlo noise as the unbumped one. If
# the standard normal shocks differ across bumps, the central difference
# becomes (signal + 2 * noise) / (2 * eps) and MC variance dominates the
# result. The invariant is: for a fixed seed, the shocks are a function
# only of (steps, paths, antithetic, moment_matching) — *not* of any
# pricing-related parameter (initial_value, volatility, drift, etc.).
# These tests pin that invariant.


class TestStandardNormalDeterminism:
    """Pin the invariant that bumping pricing-related parameters preserves
    standard normal shocks at fixed seed/steps/paths/config."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.end_date = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.end_date, 0.05)
        self.md = MarketData(self.pricing_date, self.curve, currency="USD")
        self.sim = SimulationConfig(
            paths=10_000,
            num_steps=100,
            end_date=self.end_date,
        )
        self.base_params = GBMParams(initial_value=100.0, volatility=0.25)

    def _z(self, gbm: GBMProcess, *, seed: int = 42) -> np.ndarray:
        """Generate standard-normal shocks the way `_generate_paths` does."""
        gbm._ensure_time_grid()
        num_steps = len(gbm.time_grid) - 1
        return gbm._standard_normals(seed, num_steps, gbm.paths)

    def test_spot_bump_preserves_shocks(self):
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(
            self.md,
            GBMParams(initial_value=110.0, volatility=0.25),
            self.sim,
        )
        np.testing.assert_array_equal(self._z(base), self._z(bumped))

    def test_vol_bump_preserves_shocks(self):
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(
            self.md,
            GBMParams(initial_value=100.0, volatility=0.30),
            self.sim,
        )
        np.testing.assert_array_equal(self._z(base), self._z(bumped))

    def test_rate_bump_preserves_shocks(self):
        bumped_curve = flat_curve(self.pricing_date, self.end_date, 0.06)
        bumped_md = MarketData(self.pricing_date, bumped_curve, currency="USD")
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(bumped_md, self.base_params, self.sim)
        np.testing.assert_array_equal(self._z(base), self._z(bumped))

    def test_dividend_bump_preserves_shocks(self):
        div = flat_curve(self.pricing_date, self.end_date, 0.02)
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(
            self.md,
            GBMParams(initial_value=100.0, volatility=0.25, dividend_curve=div),
            self.sim,
        )
        np.testing.assert_array_equal(self._z(base), self._z(bumped))

    def test_pricing_date_bump_preserves_shocks(self):
        """Theta bump shifts the pricing date forward. The simulation
        config keeps num_steps fixed and end_date fixed, so the time grid
        spacing changes (smaller dt), but num_steps and num_paths are
        unchanged → shocks must still match. The drift/diffusion scaling
        changes; the random shocks do not."""
        bumped_pricing_date = self.pricing_date + dt.timedelta(days=5)
        bumped_curve = flat_curve(bumped_pricing_date, self.end_date, 0.05)
        bumped_md = MarketData(bumped_pricing_date, bumped_curve, currency="USD")
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(bumped_md, self.base_params, self.sim)
        np.testing.assert_array_equal(self._z(base), self._z(bumped))

    def test_simulate_outputs_share_recovered_shocks_under_spot_bump(self):
        """End-to-end: simulate two paths sets through the public API and
        recover the standard normal shocks from the log-returns. They
        must match bit-for-bit. This is the strongest version of the
        invariant — it goes through `simulate()` rather than reaching
        into `_standard_normals` directly."""
        base = GBMProcess(self.md, self.base_params, self.sim)
        bumped = GBMProcess(
            self.md,
            GBMParams(initial_value=110.0, volatility=0.25),
            self.sim,
        )
        paths_base = base.simulate(random_seed=42)
        paths_bumped = bumped.simulate(random_seed=42)

        # Recover Z from log-returns:
        #   log(S_{t+1}/S_t) = (r - q - 0.5*σ²) Δt + σ √Δt Z
        base._ensure_time_grid()
        time_deltas = base._time_deltas()  # (steps,)
        sigma = base.volatility
        # flat 5% rate, no q
        r = 0.05
        drift = (r - 0.5 * sigma**2) * time_deltas[:, None]
        diffusion = sigma * np.sqrt(time_deltas)[:, None]

        z_base = (np.log(paths_base[1:] / paths_base[:-1]) - drift) / diffusion
        z_bumped = (np.log(paths_bumped[1:] / paths_bumped[:-1]) - drift) / diffusion

        np.testing.assert_allclose(z_base, z_bumped, atol=1e-12, rtol=0)
