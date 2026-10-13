"""Correlated multi-asset simulation on a CorrelationContext's shared grid."""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.enums import (
    AsianAveraging,
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.exceptions import UnsupportedFeatureError, ValidationError
from derivatives_pricing.market_environment import CorrelationContext, MarketData
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
    shared_time_grid,
)
from derivatives_pricing.valuation import (
    AsianSpec,
    BinomialParams,
    BarrierSpec,
    MonteCarloParams,
    OptionValuation,
    UnderlyingData,
    VanillaSpec,
    contract_event_dates,
)

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
EX_DATE = dt.datetime(2025, 5, 17)  # off the uniform monthly grid
RHO = 0.9
PATHS = 20_000


def _md() -> MarketData:
    return MarketData(PRICING_DATE, DiscountCurve.flat(0.05), currency="USD")


def _context(grid, *, paths: int = PATHS) -> CorrelationContext:
    return CorrelationContext.from_seed(
        correlation_matrix=np.array([[1.0, RHO], [RHO, 1.0]]),
        asset_names=["A", "B"],
        time_grid=grid,
        paths=paths,
        random_seed=11,
    )


def _process(ctx, name, *, dividends=None, time_grid=None, paths=PATHS) -> GBMProcess:
    return GBMProcess(
        _md(),
        GBMParams(initial_value=100.0, volatility=0.2, discrete_dividends=dividends),
        SimulationConfig(paths=paths, time_grid=ctx.time_grid if time_grid is None else time_grid),
        corr=ctx,
        name=name,
    )


class TestSharedTimeGrid:
    def test_uniform_grid_merged_with_extra_dates(self):
        early = PRICING_DATE - dt.timedelta(days=5)
        grid = shared_time_grid(PRICING_DATE, MATURITY, 12, extra_dates=[EX_DATE, early])
        assert grid[0] == PRICING_DATE and grid[-1] == MATURITY
        assert EX_DATE in grid
        assert early not in grid  # before the pricing date: dropped
        assert len(grid) == 14

    def test_matches_a_process_built_from_end_date_and_num_steps(self):
        process = GBMProcess(
            _md(),
            GBMParams(initial_value=100.0, volatility=0.2, discrete_dividends=[(EX_DATE, 1.0)]),
            SimulationConfig(paths=8, end_date=MATURITY, num_steps=12),
        )
        process.simulate(random_seed=1)
        expected = shared_time_grid(PRICING_DATE, MATURITY, 12, extra_dates=[EX_DATE])
        np.testing.assert_array_equal(process.time_grid, expected)


class TestSharedTimeGridFromInputs:
    """specs= and process_params= collect every date the grid must hold."""

    def _barrier(self) -> BarrierSpec:
        return BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=95.0,
            maturity=MATURITY,
            barrier=120.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=5,
            currency="USD",
        )

    def test_collects_event_and_ex_dividend_dates(self):
        late_ex = MATURITY + dt.timedelta(days=30)
        params_a = GBMParams(
            initial_value=100.0, volatility=0.2, discrete_dividends=[(EX_DATE, 2.5), (late_ex, 1.0)]
        )
        params_b = GBMParams(initial_value=100.0, volatility=0.3)
        barrier = self._barrier()
        grid = shared_time_grid(
            PRICING_DATE, MATURITY, 12, specs=[barrier], process_params=[params_a, params_b]
        )
        for d in contract_event_dates(barrier, PRICING_DATE):
            assert d in grid
        assert EX_DATE in grid
        assert late_ex not in grid  # after end_date: never affects the simulation
        assert grid[-1] == MATURITY

    def test_one_grid_serves_every_asset_and_contract(self):
        """The intended workflow: build the grid from the inputs, then the processes."""
        vanilla = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            currency="USD",
        )
        barrier = self._barrier()
        params_a = GBMParams(
            initial_value=100.0, volatility=0.2, discrete_dividends=[(EX_DATE, 2.5)]
        )
        params_b = GBMParams(initial_value=90.0, volatility=0.3)
        grid = shared_time_grid(
            PRICING_DATE,
            MATURITY,
            12,
            specs=[vanilla, barrier],
            process_params=[params_a, params_b],
        )
        ctx = _context(grid)
        config = SimulationConfig(paths=PATHS, time_grid=ctx.time_grid)
        a = GBMProcess(_md(), params_a, config, corr=ctx, name="A")
        b = GBMProcess(_md(), params_b, config, corr=ctx, name="B")
        for process in (a, b):
            for spec in (vanilla, barrier):
                ov = OptionValuation(process, spec, PricingMethod.MONTE_CARLO, MonteCarloParams())
                assert ov.present_value() > 0.0


class TestContractEventDates:
    def test_vanilla_is_its_maturity(self):
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            currency="USD",
        )
        assert contract_event_dates(spec, PRICING_DATE) == (MATURITY,)

    def test_discrete_barrier_adds_its_monitoring_dates(self):
        spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            barrier=120.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=4,
            currency="USD",
        )
        dates = contract_event_dates(spec, PRICING_DATE)
        assert len(dates) == 4 and dates[-1] == MATURITY

    def test_asian_drops_fixings_before_the_pricing_date(self):
        fixings = [dt.datetime(2024, 12, 1), dt.datetime(2025, 6, 1), MATURITY]
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            currency="USD",
            fixing_dates=fixings,
        )
        assert contract_event_dates(spec, PRICING_DATE) == tuple(fixings[1:])


class TestCorrelatedPaths:
    def test_increments_stay_correlated_across_an_ex_dividend_date(self):
        """Only A pays a dividend; on the shared grid every column still pairs up.

        Before the grid lived on the context, A's extra ex-date column
        shifted its draws against B's and the realised correlation of later
        increments fell from 0.9 to about 0.
        """
        grid = shared_time_grid(PRICING_DATE, MATURITY, 12, extra_dates=[EX_DATE])
        ctx = _context(grid)
        a = _process(ctx, "A", dividends=[(EX_DATE, 2.0)]).simulate()
        b = _process(ctx, "B").simulate()
        za = np.diff(np.log(a), axis=0)
        zb = np.diff(np.log(b), axis=0)
        ex_step = list(grid).index(EX_DATE) - 1  # the step ending on the ex-date
        for step in range(za.shape[0]):
            if step == ex_step:
                continue  # the dividend drop is not a diffusion increment
            rho = np.corrcoef(za[step], zb[step])[0, 1]
            assert abs(rho - RHO) < 0.02, (step, rho)


class TestProcessMustUseTheSharedGrid:
    def test_rejects_a_process_without_an_explicit_grid(self):
        ctx = _context(shared_time_grid(PRICING_DATE, MATURITY, 12))
        with pytest.raises(ValidationError, match="must simulate on the context's shared grid"):
            GBMProcess(
                _md(),
                GBMParams(initial_value=100.0, volatility=0.2),
                SimulationConfig(paths=PATHS, end_date=MATURITY, num_steps=12),
                corr=ctx,
                name="A",
            )

    def test_rejects_a_different_grid(self):
        grid = shared_time_grid(PRICING_DATE, MATURITY, 12)
        ctx = _context(grid)
        other = shared_time_grid(PRICING_DATE, MATURITY, 12, extra_dates=[EX_DATE])[:-1]
        with pytest.raises(ValidationError, match="differs from its CorrelationContext"):
            _process(ctx, "A", time_grid=other)

    def test_rejects_an_ex_dividend_date_missing_from_the_grid(self):
        ctx = _context(shared_time_grid(PRICING_DATE, MATURITY, 12))
        with pytest.raises(
            ValidationError, match="missing from its CorrelationContext.*2025-05-17"
        ):
            _process(ctx, "A", dividends=[(EX_DATE, 2.0)])

    def test_rejects_a_path_count_mismatch(self):
        ctx = _context(shared_time_grid(PRICING_DATE, MATURITY, 12))
        with pytest.raises(ValidationError, match="draws cover 20000"):
            _process(ctx, "A", paths=10_000)

    def test_rejects_a_grid_not_starting_at_the_pricing_date(self):
        later = shared_time_grid(PRICING_DATE + dt.timedelta(days=1), MATURITY, 12)
        ctx = _context(later)
        with pytest.raises(ValidationError, match="must start at the pricing date"):
            _process(ctx, "A")


class TestValuationOnACorrelatedProcess:
    def _barrier(self) -> BarrierSpec:
        return BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            barrier=130.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=5,
            currency="USD",
        )

    def test_rejects_contract_dates_missing_from_the_grid(self):
        ctx = _context(shared_time_grid(PRICING_DATE, MATURITY, 12))
        with pytest.raises(ValidationError, match="missing from the CorrelationContext"):
            OptionValuation(
                _process(ctx, "A"), self._barrier(), PricingMethod.MONTE_CARLO, MonteCarloParams()
            )

    def test_prices_when_the_grid_holds_the_contract_dates(self):
        spec = self._barrier()
        grid = shared_time_grid(PRICING_DATE, MATURITY, 12, specs=[spec])
        ctx = _context(grid)
        ov = OptionValuation(
            _process(ctx, "A"), spec, PricingMethod.MONTE_CARLO, MonteCarloParams()
        )
        assert ov.present_value() > 0.0

    def test_numerical_theta_is_rejected_but_pathwise_works(self):
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            currency="USD",
        )
        ctx = _context(shared_time_grid(PRICING_DATE, MATURITY, 12))
        ov = OptionValuation(
            _process(ctx, "A"), spec, PricingMethod.MONTE_CARLO, MonteCarloParams()
        )
        with pytest.raises(UnsupportedFeatureError, match="Numerical theta moves the pricing date"):
            ov.theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert ov.theta(greek_calc_method=GreekCalculationMethod.PATHWISE) < 0.0


class TestGridExtendingPastMaturity:
    """A shared grid usually runs past a short-dated contract's maturity.

    Every MC pricer must stop at the maturity column: the extra columns are
    simulated but must not touch the payoff, exercise or monitoring.
    """

    SHORT = dt.datetime(2025, 7, 1)

    @pytest.fixture(scope="class")
    def process(self) -> GBMProcess:
        grid = shared_time_grid(PRICING_DATE, MATURITY, 24, extra_dates=[self.SHORT])
        assert grid[-1] > self.SHORT
        ctx = CorrelationContext.from_seed(
            correlation_matrix=np.eye(2),
            asset_names=["A", "B"],
            time_grid=grid,
            paths=100_000,
            random_seed=3,
        )
        return GBMProcess(
            _md(),
            GBMParams(initial_value=100.0, volatility=0.2),
            SimulationConfig(paths=100_000, time_grid=ctx.time_grid),
            corr=ctx,
            name="A",
        )

    def _vanilla(self, option_type: OptionType, exercise: ExerciseType) -> VanillaSpec:
        return VanillaSpec(
            option_type=option_type,
            exercise_type=exercise,
            strike=100.0,
            maturity=self.SHORT,
            currency="USD",
        )

    def _assert_within_4_se(self, process, spec, reference: float) -> None:
        ov = OptionValuation(process, spec, PricingMethod.MONTE_CARLO, MonteCarloParams())
        se = ov.std_error()
        assert se is not None
        assert abs(ov.present_value() - reference) < 4.0 * se

    def _ud(self) -> UnderlyingData:
        return UnderlyingData(initial_value=100.0, volatility=0.2, market_data=_md())

    def test_european(self, process):
        spec = self._vanilla(OptionType.CALL, ExerciseType.EUROPEAN)
        reference = OptionValuation(self._ud(), spec, PricingMethod.ANALYTICAL).present_value()
        self._assert_within_4_se(process, spec, reference)

    def test_american(self, process):
        spec = self._vanilla(OptionType.PUT, ExerciseType.AMERICAN)
        reference = OptionValuation(
            self._ud(), spec, PricingMethod.BINOMIAL, BinomialParams(num_steps=2000)
        ).present_value()
        self._assert_within_4_se(process, spec, reference)

    def test_continuous_barrier(self, process):
        spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=self.SHORT,
            barrier=120.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            currency="USD",
        )
        reference = OptionValuation(self._ud(), spec, PricingMethod.ANALYTICAL).present_value()
        self._assert_within_4_se(process, spec, reference)

    def test_pathwise_greeks_use_the_maturity_column(self, process):
        spec = self._vanilla(OptionType.CALL, ExerciseType.EUROPEAN)
        mc = OptionValuation(process, spec, PricingMethod.MONTE_CARLO, MonteCarloParams())
        bsm = OptionValuation(self._ud(), spec, PricingMethod.ANALYTICAL)
        method = GreekCalculationMethod.PATHWISE
        assert np.isclose(mc.delta(greek_calc_method=method), bsm.delta(), rtol=1e-2)
        assert np.isclose(mc.vega(greek_calc_method=method), bsm.vega(), rtol=2e-2)
