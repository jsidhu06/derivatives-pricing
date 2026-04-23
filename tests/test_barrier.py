"""Tests for barrier option pricing.

Covers: BarrierSpec validation, in/out parity, analytical pricing,
initial-state handling, rebate pricing, discrete monitoring (BG correction),
and barrier Greeks across numerical, tree, and grid methods.
"""

import datetime as dt
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.exceptions import (
    ConfigurationError,
    UnsupportedFeatureError,
    ValidationError,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation import (
    BarrierSpec,
    OptionValuation,
    UnderlyingData,
    VanillaSpec,
)
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation.pde import _build_log_grid, _build_spot_grid
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams, PDEParams
from derivatives_pricing.utils import calculate_year_fraction
from helpers import flat_curve, PRICING_DATE, MATURITY, CURRENCY, SPOT, STRIKE, RATE, VOL


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_data(
    pricing_date: dt.datetime = PRICING_DATE,
    rate: float = RATE,
    maturity: dt.datetime = MATURITY,
) -> MarketData:
    curve = flat_curve(pricing_date, maturity, rate)
    return MarketData(pricing_date, curve, currency=CURRENCY)


def _underlying(
    spot: float = SPOT,
    vol: float = VOL,
    market_data: MarketData | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    if market_data is None:
        market_data = _market_data()
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_curve=dividend_curve,
    )


def _barrier_spec(
    option_type: OptionType = OptionType.CALL,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
    maturity: dt.datetime = MATURITY,
    barrier: float = 120.0,
    direction: BarrierDirection = BarrierDirection.UP,
    action: BarrierAction = BarrierAction.OUT,
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    num_observations: int | None = None,
    monitoring_dates=None,
) -> BarrierSpec:
    return BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        direction=direction,
        action=action,
        monitoring=monitoring,
        rebate=rebate,
        rebate_timing=rebate_timing,
        num_observations=num_observations,
        monitoring_dates=monitoring_dates,
    )


def _price(
    underlying: UnderlyingData | None = None,
    spec: BarrierSpec | None = None,
    **spec_kw,
) -> float:
    if underlying is None:
        underlying = _underlying()
    if spec is None:
        spec = _barrier_spec(**spec_kw)
    return OptionValuation(underlying, spec, PricingMethod.BSM).present_value()


# ===========================================================================
# BarrierSpec validation
# ===========================================================================


class TestBarrierSpecValidation:
    """Test BarrierSpec __post_init__ validation."""

    def test_valid_construction(self):
        spec = _barrier_spec()
        assert spec.option_type is OptionType.CALL
        assert spec.direction is BarrierDirection.UP
        assert spec.action is BarrierAction.OUT
        assert spec.monitoring is BarrierMonitoring.CONTINUOUS
        assert spec.rebate == 0.0
        assert spec.strike == float(STRIKE)
        assert spec.barrier == 120.0

    def test_strike_coerced_to_float(self):
        spec = _barrier_spec(strike=100)
        assert isinstance(spec.strike, float)

    def test_barrier_coerced_to_float(self):
        spec = _barrier_spec(barrier=120)
        assert isinstance(spec.barrier, float)

    def test_invalid_option_type(self):
        with pytest.raises(ConfigurationError, match="option_type"):
            BarrierSpec(
                option_type="call",  # type: ignore
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
            )

    def test_invalid_direction(self):
        with pytest.raises(ConfigurationError, match="direction"):
            BarrierSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction="up",  # type: ignore
                action=BarrierAction.OUT,
            )

    def test_invalid_action(self):
        with pytest.raises(ConfigurationError, match="action"):
            BarrierSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction=BarrierDirection.UP,
                action="out",  # type: ignore
            )

    def test_negative_strike(self):
        with pytest.raises(ValidationError, match="strike.*>= 0"):
            _barrier_spec(strike=-1.0)

    def test_non_finite_strike(self):
        with pytest.raises(ValidationError, match="strike.*finite"):
            _barrier_spec(strike=float("inf"))

    def test_barrier_must_be_positive(self):
        with pytest.raises(ValidationError, match="barrier.*> 0"):
            _barrier_spec(barrier=0.0)

    def test_negative_barrier(self):
        with pytest.raises(ValidationError, match="barrier.*> 0"):
            _barrier_spec(barrier=-10.0)

    def test_negative_rebate(self):
        with pytest.raises(ValidationError, match="rebate.*>= 0"):
            _barrier_spec(rebate=-1.0)

    def test_knock_in_rebate_at_hit_rejected(self):
        with pytest.raises(ValidationError, match="Knock-in rebate"):
            _barrier_spec(
                action=BarrierAction.IN,
                rebate=5.0,
                rebate_timing=RebateTiming.AT_HIT,
            )

    def test_knock_in_rebate_at_expiry_ok(self):
        spec = _barrier_spec(
            action=BarrierAction.IN,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert spec.rebate == 5.0

    def test_continuous_rejects_num_observations(self):
        with pytest.raises(ValidationError, match="CONTINUOUS"):
            _barrier_spec(
                monitoring=BarrierMonitoring.CONTINUOUS,
                num_observations=50,
            )

    def test_continuous_rejects_monitoring_dates(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 5)]
        with pytest.raises(ValidationError, match="CONTINUOUS"):
            _barrier_spec(
                monitoring=BarrierMonitoring.CONTINUOUS,
                monitoring_dates=dates,
            )

    def test_discrete_requires_schedule_source(self):
        with pytest.raises(ValidationError, match="exactly one"):
            _barrier_spec(monitoring=BarrierMonitoring.DISCRETE)

    def test_discrete_rejects_both_sources(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 5)]
        with pytest.raises(ValidationError, match="exactly one"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=50,
                monitoring_dates=dates,
            )

    def test_discrete_num_observations_valid(self):
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=50,
        )
        assert spec.num_observations == 50

    def test_discrete_num_observations_too_small(self):
        with pytest.raises(ValidationError, match="num_observations"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=0,
            )

    def test_discrete_num_observations_one_allowed(self):
        """N=1 is valid: a single barrier observation at maturity."""
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=1,
        )
        assert spec.num_observations == 1

    def test_discrete_monitoring_dates_valid(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 10)]
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        assert len(spec.monitoring_dates) == 9

    def test_discrete_monitoring_dates_beyond_maturity(self):
        dates = [MATURITY + dt.timedelta(days=10)]
        with pytest.raises(ValidationError, match="beyond maturity"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                monitoring_dates=dates,
            )

    def test_discrete_monitoring_dates_not_ascending(self):
        dates = [
            PRICING_DATE + dt.timedelta(days=60),
            PRICING_DATE + dt.timedelta(days=30),
        ]
        with pytest.raises(ValidationError, match="ascending"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                monitoring_dates=dates,
            )


class TestDiscreteBarrierPDEGridPlacement:
    """Discrete barrier PDE grids place the barrier between adjacent nodes."""

    def test_spot_grid_places_barrier_at_half_step(self):
        barrier = 95.0
        grid, _, _ = _build_spot_grid(
            smin=0.0,
            smax=4.0 * max(SPOT, STRIKE),
            spot_steps=200,
            anchor_spot=barrier,
            anchor_half_step=True,
        )

        left_idx = int(np.searchsorted(grid, barrier)) - 1
        assert 0 <= left_idx < grid.size - 1
        assert not np.any(np.isclose(grid, barrier, atol=1.0e-12))
        assert np.isclose(0.5 * (grid[left_idx] + grid[left_idx + 1]), barrier, atol=1.0e-12)

    @pytest.mark.parametrize("method", [PDEMethod.IMPLICIT, PDEMethod.CRANK_NICOLSON])
    def test_log_grid_places_barrier_at_half_step_in_log_space(self, method):
        barrier = 95.0
        Z, grid, _ = _build_log_grid(
            spot=SPOT,
            strike=STRIKE,
            time_to_maturity=1.0,
            volatility=VOL,
            smax_mult=4.0,
            spot_steps=200,
            time_steps=200,
            method=method,
            anchor_spot=barrier,
            anchor_half_step=True,
        )

        left_idx = int(np.searchsorted(grid, barrier)) - 1
        assert 0 <= left_idx < grid.size - 1
        assert not np.any(np.isclose(grid, barrier, atol=1.0e-12))
        assert np.isclose(np.sqrt(grid[left_idx] * grid[left_idx + 1]), barrier, atol=1.0e-12)
        assert np.isclose(0.5 * (Z[left_idx] + Z[left_idx + 1]), np.log(barrier), atol=1.0e-12)
        dz = np.diff(Z)[0]
        y_d_prime = np.log(barrier) + 0.5 * dz
        assert np.isclose(Z, y_d_prime, atol=1.0e-12).sum() == 1


# ===========================================================================
# In/Out parity: knock_in + knock_out == vanilla
# ===========================================================================


class TestBarrierInOutParity:
    """The strongest correctness check: barrier_in + barrier_out == vanilla."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    @pytest.mark.parametrize(
        "option_type,direction,barrier",
        [
            # Down barriers
            (OptionType.CALL, BarrierDirection.DOWN, 80.0),
            (OptionType.CALL, BarrierDirection.DOWN, 110.0),
            (OptionType.PUT, BarrierDirection.DOWN, 80.0),
            (OptionType.PUT, BarrierDirection.DOWN, 110.0),
            # Up barriers
            (OptionType.CALL, BarrierDirection.UP, 90.0),
            (OptionType.CALL, BarrierDirection.UP, 120.0),
            (OptionType.PUT, BarrierDirection.UP, 90.0),
            (OptionType.PUT, BarrierDirection.UP, 120.0),
        ],
    )
    def test_in_out_parity(self, option_type, direction, barrier):
        """knock_in + knock_out == vanilla for all barrier type combinations."""
        common = dict(
            option_type=option_type,
            direction=direction,
            barrier=barrier,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_in = _price(self.underlying, _barrier_spec(**common, action=BarrierAction.IN))
        pv_out = _price(self.underlying, _barrier_spec(**common, action=BarrierAction.OUT))

        vanilla_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10), (
            f"In/out parity violated: in={pv_in:.10f} + out={pv_out:.10f} "
            f"= {pv_in + pv_out:.10f} vs vanilla={pv_vanilla:.10f}"
        )

    @pytest.mark.parametrize("vol", [0.10, 0.30, 0.50])
    def test_in_out_parity_various_vols(self, vol):
        """Parity holds across different volatility levels."""
        u = _underlying(vol=vol)
        barrier = 120.0

        pv_in = _price(
            u,
            _barrier_spec(direction=BarrierDirection.UP, action=BarrierAction.IN, barrier=barrier),
        )
        pv_out = _price(
            u,
            _barrier_spec(direction=BarrierDirection.UP, action=BarrierAction.OUT, barrier=barrier),
        )

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)

    @pytest.mark.parametrize(
        "option_type,direction,barrier",
        [
            (OptionType.CALL, BarrierDirection.DOWN, 80.0),
            (OptionType.CALL, BarrierDirection.UP, 120.0),
            (OptionType.PUT, BarrierDirection.DOWN, 80.0),
            (OptionType.PUT, BarrierDirection.UP, 120.0),
        ],
    )
    @pytest.mark.parametrize(
        "pricing_method,params,rtol",
        [
            pytest.param(
                PricingMethod.PDE_FD,
                PDEParams(spot_steps=400, time_steps=400, space_grid=PDESpaceGrid.LOG_SPOT),
                2e-3,
                id="pde_fd",
            ),
            pytest.param(
                PricingMethod.BINOMIAL,
                BinomialParams(num_steps=600),
                5e-3,
                id="binomial",
            ),
        ],
    )
    def test_in_out_parity_numerical_engines(
        self, option_type, direction, barrier, pricing_method, params, rtol
    ):
        """knock_in + knock_out == vanilla on PDE_FD and Binomial engines.

        BSM parity holds exactly because of in/out complement formulas; the
        numerical engines have discretisation residual, so tolerances are
        looser than the analytical BSM parity test above.
        """
        common = dict(
            option_type=option_type,
            direction=direction,
            barrier=barrier,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_in = OptionValuation(
            self.underlying,
            _barrier_spec(**common, action=BarrierAction.IN),
            pricing_method,
            params=params,
        ).present_value()
        pv_out = OptionValuation(
            self.underlying,
            _barrier_spec(**common, action=BarrierAction.OUT),
            pricing_method,
            params=params,
        ).present_value()

        vanilla_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, pricing_method, params=params
        ).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=rtol, atol=1e-4), (
            f"In/out parity violated on {pricing_method.name}: "
            f"in={pv_in:.6f} + out={pv_out:.6f} = {pv_in + pv_out:.6f} "
            f"vs vanilla={pv_vanilla:.6f}"
        )


# ===========================================================================
# Analytical pricing — known / directional checks
# ===========================================================================


class TestBarrierAnalyticalPricing:
    """Test barrier option values for directional correctness and known properties."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_down_and_out_call_barrier_below_strike(self):
        """DOC with H < K: standard result, positive price < vanilla."""
        barrier = 80.0
        pv = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=barrier,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert pv > 0.0
        assert pv < pv_vanilla

    def test_up_and_out_call_worthless_when_barrier_below_strike(self):
        """UOC with H <= K: always worthless (barrier kills before payoff triggers)."""
        pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=90.0,  # H < K=100
        )
        assert pv == 0.0

    def test_down_and_out_put_worthless_when_barrier_above_strike(self):
        """DOP with H >= K: always worthless."""
        pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=110.0,  # H > K=100
        )
        assert pv == 0.0

    def test_up_and_in_call_equals_vanilla_when_barrier_below_strike(self):
        """UIC with H <= K: equals vanilla (barrier always hit if option has value)."""
        pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=90.0,  # H < K=100
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert np.isclose(pv, pv_vanilla, rtol=1e-10)

    def test_barrier_price_positive(self):
        """Down-and-in put with H < K produces a positive price."""
        pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
        )
        assert pv > 0.0

    def test_barrier_price_increases_with_volatility(self):
        """Higher vol → higher knock-in option prices (more likely to hit barrier)."""
        u_lo = _underlying(vol=0.15)
        u_hi = _underlying(vol=0.35)

        pv_lo = _price(
            u_lo,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            option_type=OptionType.PUT,
        )
        pv_hi = _price(
            u_hi,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            option_type=OptionType.PUT,
        )
        assert pv_hi > pv_lo

    def test_nonflat_dividend_curve(self):
        """Barrier pricing works with a non-flat dividend curve."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        div_curve = DiscountCurve.flat(0.02, end_time=T)
        u = _underlying(dividend_curve=div_curve)

        pv_in = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        pv_out = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        # In/out parity should hold with non-flat curves too
        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Initial barrier state (barrier breached at inception)
# ===========================================================================


class TestBarrierInitialState:
    """Test behavior when the barrier is already triggered at time zero."""

    def test_knock_out_up_triggered_returns_zero_no_rebate(self):
        """UOC with S >= H → knocked out at inception, PV = 0."""
        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        assert pv == 0.0

    def test_knock_out_down_triggered_returns_zero_no_rebate(self):
        """DOP with S <= H → knocked out at inception, PV = 0."""
        pv = _price(
            _underlying(spot=80.0),
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
        )
        assert pv == 0.0

    def test_knock_out_triggered_rebate_at_hit(self):
        """Knocked out at inception with rebate AT_HIT → PV = rebate (immediate)."""
        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert pv == 5.0

    def test_knock_out_triggered_rebate_at_expiry(self):
        """Knocked out at inception with rebate AT_EXPIRY → PV = rebate × df."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        df_r = float(flat_curve(PRICING_DATE, MATURITY, RATE).df(T))

        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert np.isclose(pv, 5.0 * df_r, rtol=1e-10)

    def test_knock_in_triggered_equals_vanilla(self):
        """Knock-in already triggered at inception → price as vanilla."""
        u = _underlying(spot=120.0)
        pv = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()
        assert np.isclose(pv, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Rebate pricing
# ===========================================================================


class TestBarrierRebate:
    """Test rebate legs for barrier options."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_zero_rebate_has_no_effect(self):
        """Rebate = 0 should not change the price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        assert pv_no_rebate == pv_with_rebate

    def test_knock_out_at_hit_rebate_increases_price(self):
        """Positive rebate should increase knock-out option price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_knock_out_at_expiry_rebate_increases_price(self):
        """AT_EXPIRY rebate also increases price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_knock_in_rebate_at_expiry_increases_price(self):
        """Knock-in rebate (paid at expiry if never hit) increases price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_at_hit_and_at_expiry_rebate_differ(self):
        """AT_HIT and AT_EXPIRY rebate legs produce different PVs."""
        pv_at_hit = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=10.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        pv_at_expiry = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=10.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_at_hit != pv_at_expiry
        # Both should exceed the no-rebate price
        pv_base = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        assert pv_at_hit > pv_base
        assert pv_at_expiry > pv_base


# ===========================================================================
# Discrete monitoring (Broadie-Glasserman)
# ===========================================================================


class TestBarrierDiscreteMonitoring:
    """Test Broadie-Glasserman-Kou continuity correction."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_discrete_converges_to_continuous(self):
        """As m → ∞, discrete price → continuous price."""
        continuous_pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

        # Many observations should be close to continuous
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                barrier=120.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=50000,
            ),
        )
        assert np.isclose(discrete_pv, continuous_pv, rtol=0.02)

    def test_bg_adjustment_direction_up(self):
        """UP barrier: BG shifts barrier up → knock-out price increases
        (harder to knock out with higher barrier)."""
        continuous_pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                barrier=120.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=12,
            ),
        )
        # Discrete monitoring with few observations makes knock-out worth more
        # (BG shifts barrier up — harder to knock out)
        assert discrete_pv > continuous_pv

    def test_bg_adjustment_direction_down(self):
        """DOWN barrier: BG shifts barrier down → knock-out price increases."""
        continuous_pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                option_type=OptionType.PUT,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                barrier=80.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=12,
            ),
        )
        assert discrete_pv > continuous_pv

    def test_discrete_monitoring_dates_raises_for_analytical(self):
        """Analytical engine rejects monitoring_dates (only num_observations)."""
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 10)]
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        with pytest.raises(UnsupportedFeatureError, match="num_observations"):
            OptionValuation(_underlying(), spec, PricingMethod.BSM).present_value()

    def test_in_out_parity_discrete(self):
        """In/out parity holds under BG-adjusted discrete monitoring."""
        u = self.underlying
        common = dict(
            barrier=120.0,
            direction=BarrierDirection.UP,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=50,
        )

        pv_in = _price(u, _barrier_spec(**common, action=BarrierAction.IN))
        pv_out = _price(u, _barrier_spec(**common, action=BarrierAction.OUT))

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Barrier present value against Boyle-Tian closed forms
# ===========================================================================


class TestBarrierPresentValueAgainstBoyleTianTable3:
    """Compare single-barrier PVs against Table 3 of Boyle-Tian (1998).

    Scenarios are taken from Table 3 of:

    Phelim P. Boyle & Yisong (Sam) Tian (1998) An explicit finite difference
    approach to the pricing of barrier options, Applied Mathematical Finance,
    5:1, 17-43, DOI: 10.1080/135048698334718.

    The paper's published table contains two errata relevant to these tests:
    - Table 3 uses volatility 25% (not 20%).
    - In Case IV, the second maturity is 0.5 months (not 1 month).

    We compare the library engines against the paper's closed-form column only.
    The double knock-out case is intentionally omitted.
    """

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @staticmethod
    def _paper_maturity(months: float) -> dt.datetime:
        return PRICING_DATE + dt.timedelta(days=365 * months / 12)

    @classmethod
    def _paper_underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=100.0,
            volatility=0.25,
            market_data=cls._paper_market_data(),
        )

    @classmethod
    def _paper_valuation(cls, spec: BarrierSpec, method: PricingMethod) -> OptionValuation:
        underlying = cls._paper_underlying()
        return OptionValuation(underlying, spec, method)

    _PV_CASES = [
        pytest.param(
            _barrier_spec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=95.0,
                maturity=_paper_maturity(1.0),
                barrier=97.5,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
            ),
            3.6061,
            id="table3_case_ii_1_month",
        ),
        pytest.param(
            _barrier_spec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=95.0,
                maturity=_paper_maturity(0.5),
                barrier=97.5,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
            ),
            3.7287,
            id="table3_case_ii_0_5_month",
        ),
        pytest.param(
            _barrier_spec(
                option_type=OptionType.PUT,
                exercise_type=ExerciseType.EUROPEAN,
                strike=110.0,
                maturity=_paper_maturity(1.0),
                barrier=105.0,
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
            ),
            6.7530,
            id="table3_case_iii_1_month",
        ),
        pytest.param(
            _barrier_spec(
                option_type=OptionType.PUT,
                exercise_type=ExerciseType.EUROPEAN,
                strike=110.0,
                maturity=_paper_maturity(0.5),
                barrier=105.0,
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
            ),
            7.8392,
            id="table3_case_iii_0_5_month",
        ),
        pytest.param(
            _barrier_spec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=_paper_maturity(6.0),
                barrier=90.0,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
                rebate=1.0,
                rebate_timing=RebateTiming.AT_HIT,
            ),
            8.8485,
            id="table3_case_iv_6_months_rebate",
        ),
        pytest.param(
            _barrier_spec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=_paper_maturity(0.5),
                barrier=90.0,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                monitoring=BarrierMonitoring.CONTINUOUS,
                rebate=1.0,
                rebate_timing=RebateTiming.AT_HIT,
            ),
            2.2806,
            id="table3_case_iv_0_5_month_rebate",
        ),
    ]

    # Per-engine tolerances — BSM is analytical closed-form (tightest),
    # BINOMIAL and PDE_FD carry discretization noise.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.BSM: dict(rtol=1.0e-4, atol=7.5e-4),
        PricingMethod.BINOMIAL: dict(rtol=5.0e-4, atol=2.0e-3),
        PricingMethod.PDE_FD: dict(rtol=2.0e-4, atol=1.0e-3),
    }

    @pytest.mark.parametrize("spec,paper_pv", _PV_CASES)
    def test_single_barrier_present_value_matches_paper_closed_form(
        self,
        spec: BarrierSpec,
        paper_pv: float,
        request: pytest.FixtureRequest,
    ):
        """Log all three engines side-by-side for one case, assert each."""
        engine_pvs: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BSM, PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            engine_pvs[method] = float(self._paper_valuation(spec, method).present_value())

        logger.info(
            "BT98 Table3 %s | paper=%.4f dp_bsm=%.4f dp_bn=%.4f dp_fd=%.4f",
            request.node.callspec.id,
            paper_pv,
            engine_pvs[PricingMethod.BSM],
            engine_pvs[PricingMethod.BINOMIAL],
            engine_pvs[PricingMethod.PDE_FD],
        )

        for method, pv in engine_pvs.items():
            tol = self._TOLS[method]
            assert np.isclose(pv, paper_pv, **tol), (
                f"{method.name} PV mismatch on {spec}: got {pv:.6f}, expected {paper_pv:.6f}"
            )


@pytest.mark.slow
class TestBarrierPresentValueAgainstBoyleTianTable8:
    """Compare discretely-monitored DOC PVs against Table 8 of Boyle-Tian (1998).

    Scenarios are taken from Table 8 of:

    Phelim P. Boyle & Yisong (Sam) Tian (1998) An explicit finite difference
    approach to the pricing of barrier options, Applied Mathematical Finance,
    5:1, 17-43, DOI: 10.1080/135048698334718.

    Table 8 sweeps a down-and-out call across six monitoring frequencies
    (continuous, hourly, daily, weekly, monthly, quarterly) using the
    Cheuk-Vorst (1994) convention: 1 yr = 4 q = 12 m = 52 w = 250 trading
    days = 1000 trading hours.  All discrete frequencies are scaled to the
    half-year maturity used in the paper.

    Setup: S0 = K = 100, sigma = 20%, T = 0.5 yr, r = 10%, q = 0, H = 95.
    """

    _T_YEARS = 0.5
    _PAPER_MATURITY = PRICING_DATE + dt.timedelta(days=_T_YEARS * 365)

    assert (
        calculate_year_fraction(PRICING_DATE, _PAPER_MATURITY, DayCountConvention.ACT_365F) == 0.5
    ), "Paper maturity should be exactly 0.5 years under ACT/365F"

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=100.0,
            volatility=0.20,
            market_data=cls._paper_market_data(),
        )

    @classmethod
    def _make_spec(cls, monitoring_kind: str | int) -> BarrierSpec:
        common = dict(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=cls._PAPER_MATURITY,
            barrier=95.0,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
        )
        if monitoring_kind == "continuous":
            return _barrier_spec(monitoring=BarrierMonitoring.CONTINUOUS, **common)
        return _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=int(monitoring_kind),
            **common,
        )

    # (frequency_label, monitoring_kind, paper_pv).
    # monitoring_kind: "continuous" or N obs scaled by T = 0.5 yr from
    # Cheuk-Vorst's per-year frequencies.
    _PAPER_PVS: list[tuple[str, str | int, float]] = [
        ("continuous", "continuous", 5.7163),
        ("hourly", 500, 5.8881),  # 1000 / yr
        ("daily", 125, 6.1669),  # 250  / yr
        ("weekly", 26, 6.6176),  # 52   / yr
        ("monthly", 6, 7.3100),  # 12   / yr
        ("quarterly", 2, 7.9759),  # 4    / yr
    ]

    # Per-engine tolerances, set by inspection of the actual error rates
    # against the paper.  BSM (BG continuity correction) degrades worst at
    # very low N where the asymptotic correction order matters.  PDE_FD
    # uses the Boyle-Tian half-step barrier placement which is near-exact
    # at low-to-moderate monitoring density but over-shifts slightly at
    # very dense monitoring (see pde.py module docstring); hourly is
    # granted a wider tolerance to accommodate that regime.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.BSM: dict(rtol=0.022, atol=1.0e-4),
        PricingMethod.BINOMIAL: dict(rtol=0.013, atol=1.0e-4),
        PricingMethod.PDE_FD: dict(rtol=0.001, atol=1.0e-4),
    }
    _PDE_FD_HOURLY_TOL: dict[str, float] = dict(rtol=0.015, atol=1.0e-4)

    @pytest.mark.parametrize(
        "frequency,monitoring_kind,paper_pv",
        _PAPER_PVS,
        ids=[lbl for (lbl, _, _) in _PAPER_PVS],
    )
    def test_doc_present_value_matches_paper(
        self,
        frequency: str,
        monitoring_kind: str | int,
        paper_pv: float,
    ):
        """Log all three engines side-by-side for one frequency, assert each."""
        spec = self._make_spec(monitoring_kind)
        underlying = self._paper_underlying()

        engine_pvs: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BSM, PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            engine_pvs[method] = float(OptionValuation(underlying, spec, method).present_value())

        n_obs_str = "—" if monitoring_kind == "continuous" else str(monitoring_kind)
        logger.info(
            "BT98 Table8 DOC freq=%s N=%s | paper=%.4f dp_bsm=%.4f dp_bn=%.4f dp_fd=%.4f",
            frequency,
            n_obs_str,
            paper_pv,
            engine_pvs[PricingMethod.BSM],
            engine_pvs[PricingMethod.BINOMIAL],
            engine_pvs[PricingMethod.PDE_FD],
        )

        for method, pv in engine_pvs.items():
            if method is PricingMethod.PDE_FD and frequency == "hourly":
                tol = self._PDE_FD_HOURLY_TOL
            else:
                tol = self._TOLS[method]
            assert np.isclose(pv, paper_pv, **tol), (
                f"{method.name} PV mismatch at frequency={frequency} "
                f"(N={n_obs_str}): got {pv:.6f}, expected {paper_pv:.6f}"
            )


@pytest.mark.slow
class TestBarrierPresentValueAgainstBroadieGlasserman:
    """Compare discrete DOC PVs against Broadie & Glasserman trinomial truth.

    Scenarios are taken from:

    Mark Broadie & Paul Glasserman (1997) "A Continuity Correction for
    Discrete Barrier Options", Mathematical Finance, 7(4), 325–349,
    Table 2.1.

    The paper's "True" column is computed by a trinomial procedure
    specifically modified to handle discrete barriers, and is effectively
    ground truth for a discretely-monitored DOC with m = 50 equally-spaced
    observations.

    Setup: S0 = K = 100, sigma = 30%, T = 0.2 yr, r = 10%, q = 0, m = 50.
    Barrier swept from 85 through 99 (near-the-money).

    The DP PDE_FD engine uses the Boyle-Tian half-step barrier placement,
    which brings it to near-exact agreement with _TRUTH (< 0.1% on all
    rows).  Binomial is looser due to tree-to-monitoring-date alignment
    noise, especially where the barrier approaches spot.

    BSM implements the Broadie-Glasserman-Kou closed-form correction,
    so it cannot (and is not expected to) match _TRUTH once the barrier
    approaches spot.  Instead, BSM is asserted against the paper's own
    'Corrected' column (_BGK_CORRECTED) — the value the BGK formula
    produces.
    """

    _T_YEARS = 0.2
    _PAPER_MATURITY = PRICING_DATE + dt.timedelta(days=_T_YEARS * 365)

    assert (
        calculate_year_fraction(PRICING_DATE, _PAPER_MATURITY, DayCountConvention.ACT_365F) == 0.2
    ), "Paper maturity should be exactly 0.2 years under ACT/365F"

    # barrier → paper-truth PV, trinomial reference ('True' column).
    _TRUTH: dict[int, float] = {
        85: 6.322,
        86: 6.306,
        87: 6.281,
        88: 6.242,
        89: 6.184,
        90: 6.098,
        91: 5.977,
        92: 5.810,
        93: 5.584,
        94: 5.288,
        95: 4.907,
        96: 4.427,
        97: 3.834,
        98: 3.126,
        99: 2.337,
    }

    # barrier → paper's 'Corrected' column (1): the closed-form BGK value
    # (continuous formula with barrier shifted by beta·sigma·sqrt(dt)).
    _BGK_CORRECTED: dict[int, float] = {
        85: 6.322,
        86: 6.306,
        87: 6.281,
        88: 6.242,
        89: 6.184,
        90: 6.098,
        91: 5.977,
        92: 5.810,
        93: 5.585,
        94: 5.288,
        95: 4.907,
        96: 4.428,
        97: 3.836,
        98: 3.121,
        99: 2.271,
    }

    _NUM_OBSERVATIONS = 50

    # Per-engine tolerances — calibrated from observed behaviour:
    # PDE_FD / Binomial assert against _TRUTH (they solve the discrete
    # problem).  PDE_FD hits to within ~0.1% everywhere with Boyle-Tian
    # half-step placement; Binomial drifts up to ~3% as H → S0.
    # BSM asserts against _BGK_CORRECTED (it implements the closed-form
    # asymptotic correction, not a discrete solve) — should match the
    # paper's Corrected column to the printed 3-dp precision.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.BSM: dict(rtol=0.0, atol=1.5e-3),
        PricingMethod.BINOMIAL: dict(rtol=0.035, atol=1.0e-3),
        PricingMethod.PDE_FD: dict(rtol=0.002, atol=1.0e-3),
    }

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 3.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=100.0,
            volatility=0.30,
            market_data=cls._paper_market_data(),
        )

    @classmethod
    def _make_spec(cls, barrier: float) -> BarrierSpec:
        return _barrier_spec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=cls._PAPER_MATURITY,
            barrier=barrier,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=cls._NUM_OBSERVATIONS,
        )

    @pytest.mark.parametrize(
        "barrier",
        list(_TRUTH.keys()),
        ids=[f"H_{b}" for b in _TRUTH],
    )
    def test_doc_present_value_matches_paper(self, barrier: int):
        """One barrier level → log all three engines, assert each against
        its appropriate paper reference: BSM vs Corrected, others vs True."""
        spec = self._make_spec(float(barrier))
        underlying = self._paper_underlying()

        truth = self._TRUTH[barrier]
        corrected = self._BGK_CORRECTED[barrier]

        engine_pvs: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BSM, PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            engine_pvs[method] = float(OptionValuation(underlying, spec, method).present_value())

        logger.info(
            "BG97 DOC PV H=%d m=%d | true=%.3f corrected=%.3f | dp_bsm=%.3f dp_bn=%.3f dp_fd=%.3f",
            barrier,
            self._NUM_OBSERVATIONS,
            truth,
            corrected,
            engine_pvs[PricingMethod.BSM],
            engine_pvs[PricingMethod.BINOMIAL],
            engine_pvs[PricingMethod.PDE_FD],
        )

        references: dict[PricingMethod, tuple[str, float]] = {
            PricingMethod.BSM: ("Corrected", corrected),
            PricingMethod.BINOMIAL: ("True", truth),
            PricingMethod.PDE_FD: ("True", truth),
        }
        for method, pv in engine_pvs.items():
            label, ref = references[method]
            tol = self._TOLS[method]
            assert np.isclose(pv, ref, **tol), (
                f"{method.name} PV mismatch at H={barrier} vs {label}: "
                f"got {pv:.6f}, expected {ref:.6f}"
            )


class TestBarrierDeltaAgainstBroadieGlasserman:
    """Compare discrete DOC deltas against Broadie & Glasserman Table 2.4.

    Scenarios are taken from:

    Mark Broadie & Paul Glasserman (1997) "A Continuity Correction for
    Discrete Barrier Options", Mathematical Finance, 7(4), 325-349,
    Table 2.4.

    Same setup as the PV table (S0 = K = 100, sigma = 30%, T = 0.2 yr,
    r = 10%, q = 0, m = 50, barrier 85-99) but pinning the delta at
    inception.

    Mirrors the PV test's reference strategy: Binomial and PDE_FD are
    asserted against the paper's 'True' delta column (ground truth for
    the discrete problem, obtained via extensive numerical computation);
    BSM is asserted against the paper's 'Corrected' column (the value
    the closed-form BGK correction produces), because that is what the
    BSM engine implements.
    """

    _T_YEARS = 0.2
    _PAPER_MATURITY = PRICING_DATE + dt.timedelta(days=_T_YEARS * 365)

    # barrier → paper-truth delta (BG97 Table 2.4, 'True' column).
    _TRUTH: dict[int, float] = {
        85: 0.591,
        86: 0.594,
        87: 0.600,
        88: 0.607,
        89: 0.618,
        90: 0.633,
        91: 0.653,
        92: 0.678,
        93: 0.711,
        94: 0.750,
        95: 0.798,
        96: 0.854,
        97: 0.917,
        98: 0.966,
        99: 0.958,
    }

    # barrier → paper's 'Corrected' delta (BGK closed-form value).
    _BGK_CORRECTED: dict[int, float] = {
        85: 0.591,
        86: 0.594,
        87: 0.600,
        88: 0.607,
        89: 0.618,
        90: 0.633,
        91: 0.653,
        92: 0.678,
        93: 0.710,
        94: 0.750,
        95: 0.798,
        96: 0.853,
        97: 0.917,
        98: 0.988,
        99: 1.066,
    }

    _NUM_OBSERVATIONS = 50

    # Per-engine tolerances:
    # PDE_FD / Binomial assert against _TRUTH.  PDE_FD hits truth tight
    # via Boyle-Tian half-step placement; Binomial drifts when H → S0.
    # BSM asserts against _BGK_CORRECTED — should match the paper's
    # Corrected column to the printed 3-dp precision at all 15 barriers.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.BSM: dict(rtol=0.0, atol=1.5e-3),
        PricingMethod.BINOMIAL: dict(rtol=0.025, atol=2.0e-3),
        PricingMethod.PDE_FD: dict(rtol=0.010, atol=2.0e-3),
    }

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=100.0,
            volatility=0.30,
            market_data=cls._paper_market_data(),
        )

    @classmethod
    def _make_spec(cls, barrier: float) -> BarrierSpec:
        return _barrier_spec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=cls._PAPER_MATURITY,
            barrier=barrier,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=cls._NUM_OBSERVATIONS,
        )

    @pytest.mark.parametrize(
        "barrier",
        list(_TRUTH.keys()),
        ids=[f"H_{b}" for b in _TRUTH],
    )
    def test_doc_delta_matches_paper(self, barrier: int):
        """One barrier level → log all three engines, assert each against
        its appropriate paper reference: BSM vs Corrected, others vs True."""
        spec = self._make_spec(float(barrier))
        underlying = self._paper_underlying()

        truth = self._TRUTH[barrier]
        corrected = self._BGK_CORRECTED[barrier]

        engine_deltas: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BSM, PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            engine_deltas[method] = float(OptionValuation(underlying, spec, method).delta())

        logger.info(
            "BG97 DOC Δ H=%d m=%d | true=%.3f corrected=%.3f | dp_bsm=%.3f dp_bn=%.3f dp_fd=%.3f",
            barrier,
            self._NUM_OBSERVATIONS,
            truth,
            corrected,
            engine_deltas[PricingMethod.BSM],
            engine_deltas[PricingMethod.BINOMIAL],
            engine_deltas[PricingMethod.PDE_FD],
        )

        references: dict[PricingMethod, tuple[str, float]] = {
            PricingMethod.BSM: ("Corrected", corrected),
            PricingMethod.BINOMIAL: ("True", truth),
            PricingMethod.PDE_FD: ("True", truth),
        }
        for method, delta in engine_deltas.items():
            label, ref = references[method]
            tol = self._TOLS[method]
            assert np.isclose(delta, ref, **tol), (
                f"{method.name} Δ mismatch at H={barrier} vs {label}: "
                f"got {delta:.6f}, expected {ref:.6f}"
            )


@pytest.mark.slow
class TestBarrierAmericanKIAgainstDaiKwok:
    """Compare American down-and-in call PVs against Dai & Kwok (2004).

    Scenarios are taken from:

    Min Dai & Yue Kuen Kwok (2004) "Knock-in American options",
    Journal of Futures Markets, 24(2), 179-192,
    https://doi.org/10.1002/fut.10101

    Setup: K = 100, sigma = 30%, T = 1 yr, r = 10%, q = 9%.

    The paper PVs are obtained by choosing 10,000 time steps in
    the full binomial scheme.

    The DP PDE_FD engine uses the two-surface coupled solver and matches
    the paper to ~3dp throughout.  The binomial engine uses Boyle-Lau
    retopology + the active/inactive recursion and is granted ~1% to
    accommodate cap-bind effects in the rows where the barrier sits within
    1 unit of spot.
    """

    _PAPER_MATURITY = MATURITY  # 1 year under ACT/365F (see helpers)

    assert (
        calculate_year_fraction(PRICING_DATE, _PAPER_MATURITY, DayCountConvention.ACT_365F) == 1.0
    ), "Paper maturity should be exactly 1 year under ACT/365F"

    # (barrier, spot, paper_pv) — sweeps the three regimes Dai-Kwok study.
    _PAPER_PVS: list[tuple[float, float, float]] = [
        (99.0, 99.5, 10.7432),
        (99.0, 110.5, 6.8224),
        (110.0, 110.5, 17.2062),
        (110.0, 120.5, 12.5409),
        (110.0, 140.5, 6.3553),
        (110.0, 160.5, 3.0667),
        (130.0, 130.5, 32.1285),
        (130.0, 140.5, 25.6659),
        (130.0, 150.5, 20.1773),
        (170.0, 170.5, 69.4759),
        (170.0, 180.5, 59.3874),
    ]

    # Per-engine tolerances.  PDE_FD (two-surface coupled solver) hits the
    # paper to ~3dp uniformly.  Binomial drifts more (Boyle-Lau alignment +
    # active/inactive recursion noise), so it is granted ~1%.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.PDE_FD: dict(rtol=1.0e-3, atol=2.0e-3),
        PricingMethod.BINOMIAL: dict(rtol=1.0e-2, atol=2.0e-3),
    }

    # (barrier, spot) pairs where the barrier sits within 1 unit of spot —
    # Boyle-Lau cap-bind warnings fire from the binomial engine here and
    # are expected at this near-barrier proximity.
    _BOYLE_LAU_NEAR_BARRIER: set[tuple[float, float]] = {
        (130.0, 130.5),
        (170.0, 170.5),
    }

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls, spot: float) -> UnderlyingData:
        return UnderlyingData(
            initial_value=spot,
            volatility=0.30,
            market_data=cls._paper_market_data(),
            dividend_curve=DiscountCurve.flat(0.09, 2.0),
        )

    @classmethod
    def _make_spec(cls, barrier: float) -> BarrierSpec:
        return _barrier_spec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=100.0,
            maturity=cls._PAPER_MATURITY,
            barrier=barrier,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

    @pytest.mark.parametrize(
        "barrier,spot,paper_pv",
        _PAPER_PVS,
        ids=[f"H{int(h)}_S{s}".replace(".", "_") for h, s, _ in _PAPER_PVS],
    )
    def test_american_di_call_present_value_matches_paper(
        self,
        barrier: float,
        spot: float,
        paper_pv: float,
    ):
        """Log both engines side-by-side for one (H, S), assert each against paper."""
        spec = self._make_spec(barrier)
        underlying = self._paper_underlying(spot)

        engine_pvs: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            with warnings.catch_warnings():
                if (barrier, spot) in self._BOYLE_LAU_NEAR_BARRIER:
                    warnings.filterwarnings("ignore", message=".*Boyle-Lau step alignment.*")
                engine_pvs[method] = float(
                    OptionValuation(underlying, spec, method).present_value()
                )

        logger.info(
            "DK04 AmKI H=%g S=%g | true=%.4f dp_bn=%.4f dp_fd=%.4f",
            barrier,
            spot,
            paper_pv,
            engine_pvs[PricingMethod.BINOMIAL],
            engine_pvs[PricingMethod.PDE_FD],
        )

        for method, pv in engine_pvs.items():
            tol = self._TOLS[method]
            assert np.isclose(pv, paper_pv, **tol), (
                f"{method.name} PV mismatch at H={barrier}, S={spot}: "
                f"got {pv:.6f}, expected {paper_pv:.6f}"
            )


@pytest.mark.slow
class TestAmericanBarrierDiscreteDividends:
    """Cross-engine DP check: American barrier + discrete cash dividends.

    QL's barrier engines cover only half of this combination: the FD
    engine supports discrete dividends but rejects American exercise, and
    the binomial CRR engine supports American exercise but not discrete
    dividends.  DP's binomial barrier engine is also blocked in this
    regime (see :class:`TestBinomialBarrierDiscreteDividendsGuard` and
    the guard in ``core.py``) because the escrowed-dividend tree
    adjustment smooths the ex-date spot jump into continuous drift,
    which misprices the barrier-hit probability on ex-dividend dates.

    So this test does a two-engine DP consistency check (PDE_FD vs MC)
    for all four KO/KI × direction combinations, including the
    two-surface coupled solver's ``Step C`` (dividend jump applied to
    both active and inactive surfaces, then American constraint
    re-enforced on the active surface) that underpins American
    knock-ins with discrete dividends.

    Tolerance notes:
        - PDE_FD is the reference (no structural approximation — it
          models the ex-dividend spot jump as a grid shift).
        - MC matches PDE_FD within ~2.5 % at 100k paths across both
          KO and KI scenarios; residual noise comes from GBM path
          variance plus the barrier check at discrete simulation steps.
    """

    _MAT = MATURITY  # 1 year under ACT/365F

    _R: float = 0.05
    _VOL: float = 0.20
    _SPOT: float = 100.0
    _STRIKE: float = 100.0

    # Two moderate cash dividends inside the life of the option.
    _DIVS: list[tuple[dt.datetime, float]] = [
        (PRICING_DATE + dt.timedelta(days=90), 2.0),
        (PRICING_DATE + dt.timedelta(days=270), 2.0),
    ]

    _MC_PATHS = 100_000
    _MC_NUM_STEPS = 200
    _MC_SEED = 42

    # (id, option_type, direction, action, barrier, rebate_timing)
    _SCENARIOS = [
        pytest.param(
            OptionType.CALL,
            BarrierDirection.DOWN,
            BarrierAction.OUT,
            85.0,
            RebateTiming.AT_HIT,
            id="am_down_out_call",
        ),
        pytest.param(
            OptionType.PUT,
            BarrierDirection.UP,
            BarrierAction.OUT,
            120.0,
            RebateTiming.AT_HIT,
            id="am_up_out_put",
        ),
        pytest.param(
            OptionType.CALL,
            BarrierDirection.DOWN,
            BarrierAction.IN,
            85.0,
            RebateTiming.AT_EXPIRY,
            id="am_down_in_call",
        ),
        pytest.param(
            OptionType.PUT,
            BarrierDirection.UP,
            BarrierAction.IN,
            120.0,
            RebateTiming.AT_EXPIRY,
            id="am_up_in_put",
        ),
    ]

    _TOL_MC: dict[str, float] = dict(rtol=0.025, atol=1.0e-2)

    @classmethod
    def _market_data(cls) -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(cls._R, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=cls._SPOT,
            volatility=cls._VOL,
            market_data=cls._market_data(),
            discrete_dividends=cls._DIVS,
        )

    @classmethod
    def _gbm(cls) -> GBMProcess:
        return GBMProcess(
            cls._market_data(),
            GBMParams(
                initial_value=cls._SPOT,
                volatility=cls._VOL,
                discrete_dividends=cls._DIVS,
            ),
            SimulationConfig(
                paths=cls._MC_PATHS,
                num_steps=cls._MC_NUM_STEPS,
                end_date=cls._MAT,
            ),
        )

    @classmethod
    def _make_spec(
        cls,
        option_type: OptionType,
        direction: BarrierDirection,
        action: BarrierAction,
        barrier: float,
        rebate_timing: RebateTiming,
    ) -> BarrierSpec:
        return _barrier_spec(
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
            strike=cls._STRIKE,
            maturity=cls._MAT,
            barrier=barrier,
            direction=direction,
            action=action,
            monitoring=BarrierMonitoring.CONTINUOUS,
            rebate_timing=rebate_timing,
        )

    @pytest.mark.parametrize(
        "option_type,direction,action,barrier,rebate_timing",
        _SCENARIOS,
    )
    def test_american_barrier_discrete_divs_pde_vs_mc(
        self,
        option_type: OptionType,
        direction: BarrierDirection,
        action: BarrierAction,
        barrier: float,
        rebate_timing: RebateTiming,
    ):
        """Log PDE/MC side-by-side, assert MC against PDE reference."""
        spec = self._make_spec(option_type, direction, action, barrier, rebate_timing)
        underlying = self._underlying()
        gbm = self._gbm()

        pv_pde = float(OptionValuation(underlying, spec, PricingMethod.PDE_FD).present_value())
        pv_mc = float(
            OptionValuation(
                gbm,
                spec,
                PricingMethod.MONTE_CARLO,
                params=MonteCarloParams(random_seed=self._MC_SEED),
            ).present_value()
        )

        logger.info(
            "AmBarrier+DiscDivs %s-%s %s H=%g | dp_fd=%.4f dp_mc=%.4f",
            direction.value,
            action.value,
            option_type.value,
            barrier,
            pv_pde,
            pv_mc,
        )

        assert np.isclose(pv_mc, pv_pde, **self._TOL_MC), (
            f"MC vs PDE_FD mismatch for {direction.name}-{action.name} "
            f"{option_type.name} H={barrier}: mc={pv_mc:.6f} pde={pv_pde:.6f}"
        )


class TestBinomialBarrierDiscreteDividendsGuard:
    """Binomial + barrier + discrete dividends must raise UnsupportedFeatureError.

    The escrowed-dividend tree adjustment smooths the ex-date spot jump
    into continuous drift, mispricing the barrier-hit probability on
    ex-dividend dates.  Explicit node-shift treatment would break tree
    recombination.  The guard lives in ``OptionValuation.__init__``.
    """

    def test_binomial_barrier_with_discrete_dividends_rejected(self):
        divs = [(PRICING_DATE + dt.timedelta(days=180), 2.0)]
        underlying = UnderlyingData(
            initial_value=SPOT,
            volatility=VOL,
            market_data=_market_data(),
            discrete_dividends=divs,
        )
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        with pytest.raises(UnsupportedFeatureError, match="discrete dividends"):
            OptionValuation(underlying, spec, PricingMethod.BINOMIAL)

    def test_binomial_vanilla_with_discrete_dividends_allowed(self):
        """Vanilla (non-barrier) binomial + divs remains supported."""
        divs = [(PRICING_DATE + dt.timedelta(days=180), 2.0)]
        underlying = UnderlyingData(
            initial_value=SPOT,
            volatility=VOL,
            market_data=_market_data(),
            discrete_dividends=divs,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv = OptionValuation(underlying, vanilla_spec, PricingMethod.BINOMIAL).present_value()
        assert pv > 0


# ===========================================================================
# Barrier Greeks
# ===========================================================================


class TestBarrierGreeks:
    """Test Greek computation on barrier options — NUMERICAL only."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()
        self.spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        self.val = OptionValuation(self.underlying, self.spec, PricingMethod.BSM)

    def test_delta_numerical(self):
        delta = self.val.delta()
        # Up-and-out call delta can be negative (higher spot → more likely to knock out)
        assert np.isfinite(delta)

    def test_gamma_numerical(self):
        gamma = self.val.gamma()
        assert np.isfinite(gamma)

    def test_vega_numerical(self):
        vega = self.val.vega()
        assert np.isfinite(vega)

    def test_theta_numerical(self):
        theta = self.val.theta()
        assert np.isfinite(theta)

    def test_rho_numerical(self):
        rho = self.val.rho()
        assert np.isfinite(rho)

    def test_analytical_greek_method_rejected(self):
        with pytest.raises(UnsupportedFeatureError, match="Barrier"):
            self.val.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_auto_selects_numerical(self):
        """Without explicit method, barrier Greeks auto-select NUMERICAL."""
        # This should not raise
        delta = self.val.delta()
        assert np.isfinite(delta)


@pytest.mark.slow
class TestBarrierGreeksAgainstBoyleTianTable6:
    """Compare down-and-out call Greeks against Table 6 of Boyle-Tian (1998).

    Scenarios are taken from Table 6 of:

    Phelim P. Boyle & Yisong (Sam) Tian (1998) An explicit finite difference
    approach to the pricing of barrier options, Applied Mathematical Finance,
    5:1, 17-43, DOI: 10.1080/135048698334718.

    The paper reports annualized theta. The library theta is per-day, so these
    assertions scale by 365 to compare on the same basis.
    """

    # Spot → Boyle-Tian Table 6 starred analytical (delta, gamma, theta)
    # triple. Theta is the annualized figure from the paper; library theta
    # is per-day so we scale by 365 at comparison time.
    _PAPER_GREEKS = {
        95.0: (1.1192, -0.0262, -2.6468),
        92.0: (1.2132, -0.0370, -1.1229),
        91.0: (1.2524, -0.0413, -0.5720),
        90.5: (1.2736, -0.0437, -0.2886),
        90.4: (1.2780, -0.0441, -0.2313),
        90.3: (1.2825, -0.0446, -0.1738),
        90.2: (1.2869, -0.0451, -0.1161),
    }

    # Binomial tree greeks are known to degrade very close to the barrier
    # (Boyle-Lau retopology + discrete-grid noise).  Regression is enforced
    # at those spots via BSM numerical and PDE grid greeks only.
    _BINOMIAL_SKIP_SPOTS = {90.5, 90.4, 90.3, 90.2}

    # Per-engine tolerances for each greek.
    _TOLS = {
        "delta": {
            PricingMethod.BSM: dict(rtol=1.0e-2, atol=1.0e-2),
            PricingMethod.BINOMIAL: dict(rtol=1.0e-2, atol=1.0e-2),
            PricingMethod.PDE_FD: dict(rtol=1.0e-2, atol=1.0e-2),
        },
        "gamma": {
            PricingMethod.BSM: dict(rtol=1.5e-2, atol=5.0e-4),
            PricingMethod.BINOMIAL: dict(rtol=1.5e-2, atol=5.0e-4),
            PricingMethod.PDE_FD: dict(rtol=1.5e-2, atol=5.0e-4),
        },
        "theta": {
            PricingMethod.BSM: dict(rtol=2.0e-2, atol=3.0e-2),
            PricingMethod.BINOMIAL: dict(rtol=2.0e-2, atol=3.0e-2),
            PricingMethod.PDE_FD: dict(rtol=2.0e-2, atol=3.0e-2),
        },
    }

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls, spot: float) -> UnderlyingData:
        return UnderlyingData(
            initial_value=spot,
            volatility=0.25,
            market_data=cls._paper_market_data(),
            dividend_curve=DiscountCurve.flat(0.0, 2.0),
        )

    @staticmethod
    def _paper_spec() -> BarrierSpec:
        return BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=PRICING_DATE + dt.timedelta(days=365),
            barrier=90.0,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

    @classmethod
    def _engine_greek(cls, spot: float, method: PricingMethod, greek: str) -> float:
        """Return a single greek (delta/gamma/theta) from the given engine.

        Theta is annualized (×365) to match the paper's reporting basis.
        """
        valuation = OptionValuation(cls._paper_underlying(spot), cls._paper_spec(), method)
        value = float(getattr(valuation, greek)())
        if greek == "theta":
            value *= 365.0
        return value

    @pytest.mark.parametrize(
        "spot",
        list(_PAPER_GREEKS.keys()),
        ids=[f"spot_{s:.1f}".replace(".", "_") for s in _PAPER_GREEKS],
    )
    @pytest.mark.parametrize("greek", ["delta", "gamma", "theta"])
    def test_down_and_out_call_greek_matches_paper(self, spot: float, greek: str):
        """One (spot, greek) → log all three engines, assert each separately."""
        paper_value = self._PAPER_GREEKS[spot][["delta", "gamma", "theta"].index(greek)]

        engine_values: dict[PricingMethod, float | None] = {}
        for method in (PricingMethod.BSM, PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            if method is PricingMethod.BINOMIAL and spot in self._BINOMIAL_SKIP_SPOTS:
                engine_values[method] = None
                continue
            engine_values[method] = self._engine_greek(spot, method, greek)

        def _fmt(v: float | None) -> str:
            return f"{v:.6f}" if v is not None else "skipped"

        logger.info(
            "BT98 Table6 DOC spot=%.2f %s | paper=%.6f dp_bsm=%s dp_bn=%s dp_fd=%s",
            spot,
            greek,
            paper_value,
            _fmt(engine_values[PricingMethod.BSM]),
            _fmt(engine_values[PricingMethod.BINOMIAL]),
            _fmt(engine_values[PricingMethod.PDE_FD]),
        )

        tol = self._TOLS[greek]
        for method, value in engine_values.items():
            if value is None:
                continue
            assert np.isclose(value, paper_value, **tol[method]), (
                f"{greek} mismatch for {method.name} at spot={spot}: "
                f"got {value:.6f}, expected {paper_value:.6f}"
            )


@pytest.mark.slow
class TestBarrierGreeksAgainstBoyleTianTable9:
    """Compare discretely-monitored DOC Greeks against Table 9 of Boyle-Tian (1998).

    Scenarios are taken from Table 9 of:

    Phelim P. Boyle & Yisong (Sam) Tian (1998) An explicit finite difference
    approach to the pricing of barrier options, Applied Mathematical Finance,
    5:1, 17-43, DOI: 10.1080/135048698334718.

    Table 9 sweeps Greeks (delta, gamma, theta) for a down-and-out call
    across five monitoring frequencies — continuously, daily, weekly,
    monthly, quarterly — using the Cheuk-Vorst (1994) convention:
    1 yr = 4 q = 12 m = 52 w = 250 trading days.  All discrete frequencies
    are scaled to the half-year maturity used in the paper.

    Setup: S0 = K = 100, sigma = 20%, T = 0.5 yr, r = 10%, q = 0, H = 95.
    The paper reports annualized theta; the library theta is per-day, so
    the comparison scales by 365.
    """

    _T_YEARS = 0.5
    _PAPER_MATURITY = PRICING_DATE + dt.timedelta(days=_T_YEARS * 365)

    # frequency_label → (monitoring_kind, (delta, gamma, theta_annual))
    # monitoring_kind: "continuous" or N obs scaled by T = 0.5 yr.
    _PAPER_GREEKS: dict[str, tuple[str | int, tuple[float, float, float]]] = {
        "continuous": ("continuous", (1.0474, -0.0272, -4.4572)),
        "daily": (125, (0.9895, -0.0208, -5.1264)),  # 250 / yr
        "weekly": (26, (0.9312, -0.0134, -5.9688)),  # 52  / yr
        "monthly": (6, (0.8055, 0.0155, -10.4166)),  # 12  / yr
        "quarterly": (2, (0.6955, 0.0250, -11.1626)),  # 4   / yr
    }

    # Per-engine tolerances.  BSM's Broadie-Glasserman-Kou continuity
    # correction is an asymptotic series in 1/√N; when differentiated into
    # greeks at small N, errors grow large (even sign flips for gamma at
    # monthly/quarterly).  BSM is therefore excluded from this test
    # entirely. The Binomial gamma
    # tolerance at weekly (N=26) accommodates tree-to-monitoring-date
    # alignment noise where 5000 default tree steps spread over 26
    # non-uniform observation dates leaves sub-integer rounding errors
    # that amplify in the second derivative.
    _TOLS: dict[str, dict[PricingMethod, dict[str, float]]] = {
        "delta": {
            PricingMethod.BINOMIAL: dict(rtol=1.5e-2, atol=1.0e-3),
            PricingMethod.PDE_FD: dict(rtol=1.5e-2, atol=1.0e-3),
        },
        "gamma": {
            PricingMethod.BINOMIAL: dict(rtol=1.5e-1, atol=2.0e-3),
            PricingMethod.PDE_FD: dict(rtol=5.0e-2, atol=2.0e-3),
        },
        "theta": {
            PricingMethod.BINOMIAL: dict(rtol=7.0e-2, atol=2.0e-2),
            PricingMethod.PDE_FD: dict(rtol=3.0e-2, atol=2.0e-2),
        },
    }

    @staticmethod
    def _paper_market_data() -> MarketData:
        return MarketData(
            PRICING_DATE,
            DiscountCurve.flat(0.10, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _paper_underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=100.0,
            volatility=0.20,
            market_data=cls._paper_market_data(),
        )

    @classmethod
    def _make_spec(cls, monitoring_kind: str | int) -> BarrierSpec:
        common = dict(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=cls._PAPER_MATURITY,
            barrier=95.0,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
        )
        if monitoring_kind == "continuous":
            return _barrier_spec(monitoring=BarrierMonitoring.CONTINUOUS, **common)
        return _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=int(monitoring_kind),
            **common,
        )

    @classmethod
    def _engine_greek(
        cls,
        monitoring_kind: str | int,
        method: PricingMethod,
        greek: str,
    ) -> float:
        """Return a single greek from the given engine; theta annualized ×365."""
        valuation = OptionValuation(
            cls._paper_underlying(), cls._make_spec(monitoring_kind), method
        )
        value = float(getattr(valuation, greek)())
        if greek == "theta":
            value *= 365.0
        return value

    @pytest.mark.parametrize(
        "frequency",
        list(_PAPER_GREEKS.keys()),
        ids=list(_PAPER_GREEKS.keys()),
    )
    @pytest.mark.parametrize("greek", ["delta", "gamma", "theta"])
    def test_doc_greek_matches_paper(self, frequency: str, greek: str):
        """One (frequency, greek) → log Binomial + PDE_FD, assert each."""
        monitoring_kind, triple = self._PAPER_GREEKS[frequency]
        paper_value = triple[["delta", "gamma", "theta"].index(greek)]

        engine_values: dict[PricingMethod, float] = {}
        for method in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            engine_values[method] = self._engine_greek(monitoring_kind, method, greek)

        n_obs_str = "—" if monitoring_kind == "continuous" else str(monitoring_kind)
        logger.info(
            "BT98 Table9 DOC freq=%s N=%s %s | paper=%.6f dp_bn=%.6f dp_fd=%.6f",
            frequency,
            n_obs_str,
            greek,
            paper_value,
            engine_values[PricingMethod.BINOMIAL],
            engine_values[PricingMethod.PDE_FD],
        )

        tol = self._TOLS[greek]
        for method, value in engine_values.items():
            assert np.isclose(value, paper_value, **tol[method]), (
                f"{greek} mismatch for {method.name} at frequency={frequency} "
                f"(N={n_obs_str}): got {value:.6f}, expected {paper_value:.6f}"
            )


# ===========================================================================
# Binomial barrier NUMERICAL guard (Boyle-Lau retopology)
# ===========================================================================
# Bumping spot/vol/T on a binomial barrier valuation re-runs Boyle-Lau
# alignment with new inputs, which can pick a different num_steps for each
# bumped tree. The resulting central difference compares two unrelated tree
# topologies, so we explicitly block NUMERICAL bump-and-revalue greeks on
# binomial barriers (rho excepted — the Boyle-Lau formula has no r term).


class TestBinomialBarrierNumericalGuard:
    """Regression tests for the binomial-barrier NUMERICAL guard policy."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()
        self.spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        self.val = OptionValuation(
            self.underlying,
            self.spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=200),
        )

    @pytest.mark.parametrize("greek_name", ["delta", "gamma", "theta", "vega"])
    def test_explicit_numerical_blocked(self, greek_name):
        """Explicit NUMERICAL on delta/gamma/theta/vega raises."""
        fn = getattr(self.val, greek_name)
        with pytest.raises(UnsupportedFeatureError, match="NUMERICAL"):
            fn(greek_calc_method=GreekCalculationMethod.NUMERICAL)

    def test_explicit_numerical_rho_allowed(self):
        """Explicit NUMERICAL on rho is allowed (rate bumps don't enter
        the Boyle-Lau formula, so the bumped trees share the same topology)."""
        rho = self.val.rho(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isfinite(rho)

    def test_auto_select_vega_blocked(self):
        """vega() with no explicit method routes to NUMERICAL (no tree-
        native path) and the guard fires from the auto-select branch."""
        with pytest.raises(UnsupportedFeatureError, match="NUMERICAL"):
            self.val.vega()

    @pytest.mark.parametrize("greek_name", ["delta", "gamma", "theta"])
    def test_auto_select_picks_tree(self, greek_name):
        """delta/gamma/theta with no explicit method auto-select TREE on
        binomial, bypassing the NUMERICAL guard entirely."""
        fn = getattr(self.val, greek_name)
        value = fn()
        assert np.isfinite(value)

    def test_auto_select_rho_uses_numerical(self):
        """rho() has no tree-native path so it auto-selects NUMERICAL,
        but rho is exempt from the guard so this works."""
        rho = self.val.rho()
        assert np.isfinite(rho)


# ===========================================================================
# Binomial barrier coverage
# ===========================================================================


class TestBinomialBarrierCoverage:
    """Cover binomial barrier paths not exercised elsewhere."""

    def test_discrete_monitoring_dates_explicit(self):
        """Binomial barrier with explicit monitoring_dates (not num_observations)."""
        dates = [PRICING_DATE + dt.timedelta(days=d) for d in (30, 60, 90, 120, 150)]
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        val = OptionValuation(
            _underlying(), spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=200)
        )
        pv = val.present_value()
        assert pv > 0

    def test_boyle_lau_cap_bind_warning(self):
        """When barrier is very close to spot, Boyle-Lau cap-bind warning fires."""
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=100.01,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = OptionValuation(
                _underlying(spot=100.0),
                spec,
                PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=50),
            )
            val.present_value()
        assert any("Boyle-Lau step alignment" in str(warning.message) for warning in w)

    def test_knock_in_triggered_at_inception_binomial_pv(self):
        """KI triggered at inception via binomial → matches vanilla binomial."""
        u = _underlying(spot=120.0)
        params = BinomialParams(num_steps=200)

        ki_spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        pv_ki = OptionValuation(u, ki_spec, PricingMethod.BINOMIAL, params=params).present_value()

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            u, vanilla_spec, PricingMethod.BINOMIAL, params=params
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-10)

    def test_knock_out_triggered_at_inception_rebate_at_hit_binomial(self):
        """KO triggered at inception with AT_HIT rebate via binomial."""
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        pv = OptionValuation(
            _underlying(spot=120.0), spec, PricingMethod.BINOMIAL, params=BinomialParams()
        ).present_value()
        assert pv == 5.0


# ===========================================================================
# Unsupported feature errors
# ===========================================================================


class TestBarrierUnsupported:
    """Test that unsupported configurations raise appropriate errors."""

    def test_american_exercise_not_supported(self):
        spec = _barrier_spec(exercise_type=ExerciseType.AMERICAN)
        with pytest.raises(UnsupportedFeatureError, match="AMERICAN"):
            OptionValuation(_underlying(), spec, PricingMethod.BSM)

    def test_discrete_dividends_rejected(self):
        divs = [(PRICING_DATE + dt.timedelta(days=90), 2.0)]
        u = UnderlyingData(
            initial_value=SPOT,
            volatility=VOL,
            market_data=_market_data(),
            discrete_dividends=divs,
        )
        spec = _barrier_spec()
        with pytest.raises(UnsupportedFeatureError, match="discrete dividends"):
            OptionValuation(u, spec, PricingMethod.BSM).present_value()


# ===========================================================================
# Monte Carlo barrier coverage
# ===========================================================================

_MC_SEED = MonteCarloParams(random_seed=42)
_MC_PATHS = 50_000
_MC_STEPS = 200

# ===========================================================================
# Discrete monitoring — BG-corrected analytical vs MC cross-check
# ===========================================================================
# The BSM discrete-monitoring path applies the Broadie-Glasserman-Kou
# continuity correction to the continuous closed-form (shift the effective
# barrier by β·σ·√Δt). The MC path explicitly checks the barrier at each
# monitoring date on simulated paths — structurally unrelated to BG. If the
# BG formula is wrong, these two will disagree.


def _mc_gbm(
    spot: float = SPOT,
    vol: float = VOL,
    market_data: MarketData | None = None,
    dividend_curve: DiscountCurve | None = None,
    paths: int = _MC_PATHS,
    num_steps: int = _MC_STEPS,
) -> GBMProcess:
    if market_data is None:
        market_data = _market_data()
    return GBMProcess(
        market_data,
        GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve),
        SimulationConfig(paths=paths, num_steps=num_steps, end_date=MATURITY),
    )


def _mc_price(
    gbm: GBMProcess | None = None,
    spec: BarrierSpec | None = None,
    params: MonteCarloParams = _MC_SEED,
    **spec_kw,
) -> float:
    if gbm is None:
        gbm = _mc_gbm()
    if spec is None:
        spec = _barrier_spec(**spec_kw)
    return OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=params).present_value()


class TestBarrierDiscreteBGvsMC:
    """Cross-validate the BG continuity correction against pathwise MC."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "option_type,direction,action,barrier,num_observations",
        [
            pytest.param(
                OptionType.CALL,
                BarrierDirection.UP,
                BarrierAction.OUT,
                120.0,
                60,
                id="up_out_call_monthly",
            ),
            pytest.param(
                OptionType.PUT,
                BarrierDirection.DOWN,
                BarrierAction.OUT,
                80.0,
                60,
                id="down_out_put_monthly",
            ),
            pytest.param(
                OptionType.CALL,
                BarrierDirection.UP,
                BarrierAction.IN,
                120.0,
                50,
                id="up_in_call_biweekly",
            ),
        ],
    )
    def test_bg_analytical_matches_mc_pathwise(
        self, option_type, direction, action, barrier, num_observations
    ):
        """BSM (BG-corrected analytical) ≈ MC (pathwise checks) for
        discrete-monitoring barriers. The two implementations are
        structurally unrelated — agreement validates the BG formula."""
        spec = _barrier_spec(
            option_type=option_type,
            direction=direction,
            action=action,
            barrier=barrier,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=num_observations,
        )
        pv_bsm = OptionValuation(_underlying(), spec, PricingMethod.BSM).present_value()

        gbm = _mc_gbm(num_steps=max(200, num_observations * 10))
        pv_mc = OptionValuation(
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, log_timings=True),
        ).present_value()

        # MC noise + BG approximation residual. BG is O((Δt_obs)^(3/2))
        # accurate — for monthly monitoring with σ=0.20 the residual can
        # reach a few percent on its own, with MC noise on top. ~5% is a
        # meaningful bound: a broken BG formula would miss by >10%.
        assert np.isclose(pv_bsm, pv_mc, rtol=0.05, atol=5e-3), (
            f"BG vs MC mismatch: BSM={pv_bsm:.6f} MC={pv_mc:.6f}"
        )


_EXERCISE_TYPES = [ExerciseType.EUROPEAN, ExerciseType.AMERICAN]


@pytest.mark.parametrize("exercise_type", _EXERCISE_TYPES)
class TestBarrierMCInceptionHit:
    """MC paths where the barrier is already triggered at time zero.

    Inception-hit semantics are identical for European and American exercise
    (option is dead/active at t=0; no exercise decisions yet matter), so each
    test is parametrized over both exercise types.
    """

    # -- Continuous monitoring --

    def test_continuous_ko_inception_pv_zero(self, exercise_type):
        """Continuous UOC with S >= H → MC weight = 0 → PV = 0."""
        pv = _mc_price(
            _mc_gbm(spot=120.0),
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        assert np.isclose(pv, 0.0, atol=1e-10)

    def test_continuous_ko_inception_rebate_at_hit(self, exercise_type):
        """Continuous UOC at inception with AT_HIT rebate → PV = rebate."""
        pv = _mc_price(
            _mc_gbm(spot=120.0),
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert np.isclose(pv, 5.0, atol=1e-10)

    def test_continuous_ko_inception_rebate_at_expiry(self, exercise_type):
        """Continuous UOC at inception with AT_EXPIRY rebate → PV = rebate * df."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        df = float(flat_curve(PRICING_DATE, MATURITY, RATE).df(T))

        pv = _mc_price(
            _mc_gbm(spot=120.0),
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert np.isclose(pv, 5.0 * df, rtol=1e-6)

    def test_continuous_ki_inception_equals_vanilla(self, exercise_type):
        """Continuous UIC with S >= H → knocked in at inception → vanilla."""
        gbm = _mc_gbm(spot=120.0)
        pv_ki = _mc_price(
            gbm,
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=exercise_type,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-6)

    # -- Discrete monitoring --

    def test_discrete_ko_inception_pv_zero(self, exercise_type):
        """Discrete DOP with S <= H and pricing date in the schedule → PV = 0.

        The library's default ``num_observations`` schedule excludes the
        pricing date (BGK convention), so the inception-hit code path only
        fires when ``monitoring_dates`` explicitly includes it.  Here we
        pass a schedule anchored at the pricing date.
        """
        dates = pd.date_range(PRICING_DATE, MATURITY, periods=12).to_pydatetime().tolist()
        pv = _mc_price(
            _mc_gbm(spot=80.0),
            option_type=OptionType.PUT,
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        assert np.isclose(pv, 0.0, atol=1e-10)

    def test_discrete_ki_inception_equals_vanilla_aligned(self, exercise_type):
        """Discrete DIC at inception with grid-aligned pricing-date observation.

        Passing ``monitoring_dates`` that include the pricing date AND land
        exactly on the simulation grid ensures both that the inception-hit
        code path fires and that no extra dates are injected into the
        simulation grid — so the random draws are identical to vanilla.
        """
        gbm = _mc_gbm(spot=80.0)
        # Build monitoring dates directly from the simulation grid so they
        # align exactly; take N+1 points starting at pricing_date.
        gbm.simulate(random_seed=42)  # force time_grid to materialize
        dates = (
            pd.date_range(PRICING_DATE, MATURITY, periods=_MC_STEPS + 1).to_pydatetime().tolist()
        )
        pv_ki = _mc_price(
            gbm,
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=exercise_type,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-10)

    def test_discrete_ki_inception_equals_vanilla_unaligned(self, exercise_type):
        """Discrete DIC at inception with non-aligned pricing-date observation.

        Monitoring dates include pricing_date but don't align with the
        simulation grid, so extra dates get injected — random draws differ
        from vanilla.  Both prices converge to the same expectation; we
        compare within MC noise.
        """
        dates = pd.date_range(PRICING_DATE, MATURITY, periods=12).to_pydatetime().tolist()
        gbm = _mc_gbm(spot=80.0)
        pv_ki = _mc_price(
            gbm,
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=exercise_type,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=0.01)


@pytest.mark.parametrize("exercise_type", _EXERCISE_TYPES)
class TestBarrierMCDiscreteRebate:
    """Discrete monitoring rebate paths in MC."""

    def test_discrete_ko_rebate_at_hit_positive(self, exercise_type):
        """Discrete KO with rebate AT_HIT: PV should include rebate component."""
        pv_no_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_discrete_ko_rebate_at_expiry_positive(self, exercise_type):
        """Discrete KO with rebate AT_EXPIRY: PV should include rebate component."""
        pv_no_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_discrete_ki_rebate_at_expiry_positive(self, exercise_type):
        """Discrete KI with rebate AT_EXPIRY: never-knocked-in paths receive rebate."""
        pv_no_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate


@pytest.mark.parametrize("exercise_type", _EXERCISE_TYPES)
class TestBarrierMCContinuousRebateAtExpiry:
    """Continuous monitoring KO rebate AT_EXPIRY path in MC."""

    def test_continuous_ko_rebate_at_expiry_positive(self, exercise_type):
        """Continuous KO with AT_EXPIRY rebate: PV includes discounted rebate."""
        pv_no_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
        )
        pv_with_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_continuous_ki_rebate_at_expiry_positive(self, exercise_type):
        """Continuous KI with AT_EXPIRY rebate: never-in paths receive rebate."""
        pv_no_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        pv_with_rebate = _mc_price(
            exercise_type=exercise_type,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate


class TestBarrierMCNonBarrierAwareBasis:
    """Test the non-barrier-aware LSM regression path."""

    def test_american_ko_without_barrier_aware_basis(self):
        """American KO with barrier_aware_basis=False should still price reasonably."""
        params_aware = MonteCarloParams(random_seed=42, barrier_aware_basis=True)
        params_naive = MonteCarloParams(random_seed=42, barrier_aware_basis=False)

        spec = _barrier_spec(
            exercise_type=ExerciseType.AMERICAN,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=85.0,
        )
        gbm = _mc_gbm()

        pv_aware = OptionValuation(
            gbm, spec, PricingMethod.MONTE_CARLO, params=params_aware
        ).present_value()
        pv_naive = OptionValuation(
            gbm, spec, PricingMethod.MONTE_CARLO, params=params_naive
        ).present_value()

        # Both should be positive and in the same ballpark
        assert pv_aware > 0
        assert pv_naive > 0
        assert np.isclose(pv_naive, pv_aware, rtol=0.05), (
            f"barrier_aware_basis=False ({pv_naive:.4f}) vs True ({pv_aware:.4f})"
        )
