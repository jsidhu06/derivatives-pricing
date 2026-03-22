"""Integration-style tests for ``PayoffSpec`` custom payoff contracts.

Scope:
- Present value sanity checks for non-trivial custom payoffs (binomial, MC, PDE)
- American-vs-European monotonicity for engines that support both styles
- Greek behavior for PayoffSpec:
    - Numerical bump-and-revalue Greeks are supported
    - Binomial TREE extraction (delta/gamma/theta) is compared to numerical
    - MC PATHWISE and LIKELIHOOD_RATIO methods are rejected for non-vanilla specs
- Engine guardrails:
    - BSM rejects PayoffSpec (requires vanilla CALL/PUT)
- PDE custom payoff pricing:
    - Affine boundary-model conditions: inferred and explicit
    - Cross-engine consistency (PDE vs binomial / MC)
- Cross-checks against replication / cross-engine consistency where appropriate
"""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import (
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.exceptions import (
    UnsupportedFeatureError,
    ValidationError,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from derivatives_pricing.valuation import (
    BinomialParams,
    MonteCarloParams,
    OptionValuation,
    PayoffSpec,
    PayoffBoundaryModel,
    WingBoundary,
    PDEParams,
    UnderlyingData,
    VanillaSpec,
)
from helpers import flat_curve

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"
SPOT = 100.0
RATE = 0.05
VOL = 0.20
MC_SEED = 42


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------


def _capped_strangle(S):
    """Capped strangle: long 90 put + long 110 call, capped at 40."""
    return np.minimum(40.0, np.maximum(90.0 - S, 0) + np.maximum(S - 110.0, 0))


def _bull_call_spread(S):
    """Bull call spread: long 95 call, short 115 call."""
    return np.maximum(S - 95.0, 0) - np.maximum(S - 115.0, 0)


def _digital_call(S):
    """Cash-or-nothing digital call with strike 100, payout 1."""
    return np.where(S > 100.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curve() -> DiscountCurve:
    return flat_curve(PRICING_DATE, MATURITY, RATE)


def _md() -> MarketData:
    return MarketData(PRICING_DATE, _curve(), currency=CURRENCY)


def _ud(**overrides) -> UnderlyingData:
    kw = dict(initial_value=SPOT, volatility=VOL, market_data=_md())
    kw.update(overrides)
    return UnderlyingData(**kw)


def _gbm(**overrides) -> GBMProcess:
    md = _md()
    params = GBMParams(
        initial_value=overrides.get("initial_value", SPOT),
        volatility=overrides.get("volatility", VOL),
    )
    sim_config = SimulationConfig(
        paths=100_000,
        num_steps=52,
        end_date=MATURITY,
    )
    return GBMProcess(md, params, sim_config)


def _payoff_spec(
    payoff_fn,
    exercise_type=ExerciseType.EUROPEAN,
) -> PayoffSpec:
    return PayoffSpec(
        exercise_type=exercise_type,
        maturity=MATURITY,
        payoff_fn=payoff_fn,
        currency=CURRENCY,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Present value — smoke tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecPresentValue:
    """Test that custom payoffs produce sensible present values."""

    @pytest.mark.parametrize(
        "payoff_fn,label",
        [
            (_capped_strangle, "capped_strangle"),
            (_bull_call_spread, "bull_call_spread"),
            (_digital_call, "digital_call"),
        ],
    )
    def test_binomial_pv_positive(self, payoff_fn, label):
        """Binomial PV for non-trivial custom payoffs should be positive."""
        spec = _payoff_spec(payoff_fn)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        pv = ov.present_value()
        assert pv > 0, f"{label}: expected positive PV, got {pv}"
        assert np.isfinite(pv)

    @pytest.mark.parametrize(
        "payoff_fn,label",
        [
            (_capped_strangle, "capped_strangle"),
            (_bull_call_spread, "bull_call_spread"),
            (_digital_call, "digital_call"),
        ],
    )
    def test_mc_pv_positive(self, payoff_fn, label):
        """MC PV for non-trivial custom payoffs should be positive."""
        spec = _payoff_spec(payoff_fn)
        params = MonteCarloParams(random_seed=MC_SEED)
        gbm = _gbm()
        ov = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=params)
        pv = ov.present_value()
        assert pv > 0, f"{label}: expected positive PV, got {pv}"
        assert np.isfinite(pv)

    def test_binomial_mc_cross_engine_consistency(self):
        """Binomial and MC should produce close PVs for the same custom payoff."""
        spec = _payoff_spec(_bull_call_spread)

        binom_params = BinomialParams(num_steps=500)
        binom_pv = OptionValuation(
            _ud(), spec, PricingMethod.BINOMIAL, params=binom_params
        ).present_value()

        mc_params = MonteCarloParams(random_seed=MC_SEED)
        mc_pv = OptionValuation(
            _gbm(), spec, PricingMethod.MONTE_CARLO, params=mc_params
        ).present_value()

        assert np.isclose(binom_pv, mc_pv, rtol=0.03), f"binomial={binom_pv:.4f} vs MC={mc_pv:.4f}"

    def test_bull_spread_bounded_by_max_spread(self):
        """Bull call spread PV must be <= discounted max spread (20)."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=500)
        pv = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params).present_value()
        ttm = 1.0  # approx
        df = np.exp(-RATE * ttm)
        assert pv <= 20.0 * df + 0.01  # small tolerance

    def test_capped_strangle_bounded_by_cap(self):
        """Capped strangle PV must be <= discounted cap (40)."""
        spec = _payoff_spec(_capped_strangle)
        params = BinomialParams(num_steps=500)
        pv = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params).present_value()
        ttm = 1.0
        df = np.exp(-RATE * ttm)
        assert pv <= 40.0 * df + 0.01


# ═══════════════════════════════════════════════════════════════════════════
# American exercise
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecAmerican:
    """American PayoffSpec should produce >= European PV."""

    def test_american_ge_european_binomial(self):
        euro_spec = _payoff_spec(_capped_strangle, ExerciseType.EUROPEAN)
        amer_spec = _payoff_spec(_capped_strangle, ExerciseType.AMERICAN)
        params = BinomialParams(num_steps=300)

        euro_pv = OptionValuation(
            _ud(), euro_spec, PricingMethod.BINOMIAL, params=params
        ).present_value()
        amer_pv = OptionValuation(
            _ud(), amer_spec, PricingMethod.BINOMIAL, params=params
        ).present_value()

        assert amer_pv >= euro_pv - 1e-8

    def test_american_ge_european_mc(self):
        euro_spec = _payoff_spec(_bull_call_spread, ExerciseType.EUROPEAN)
        amer_spec = _payoff_spec(_bull_call_spread, ExerciseType.AMERICAN)
        mc_params = MonteCarloParams(random_seed=MC_SEED)

        euro_pv = OptionValuation(
            _gbm(), euro_spec, PricingMethod.MONTE_CARLO, params=mc_params
        ).present_value()
        amer_pv = OptionValuation(
            _gbm(), amer_spec, PricingMethod.MONTE_CARLO, params=mc_params
        ).present_value()

        assert amer_pv >= euro_pv - 1e-8


# ═══════════════════════════════════════════════════════════════════════════
# Greeks — numerical bump-and-revalue
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecGreeks:
    """Test numerical Greeks for PayoffSpec."""

    def test_bull_spread_delta_positive(self):
        """Bull call spread delta should be positive (long exposure)."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        delta = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert delta > 0

    def test_bull_spread_delta_bounded(self):
        """Bull call spread delta must be in [0, 1]."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        delta = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert 0 <= delta <= 1 + 1e-6

    def test_capped_strangle_gamma_positive_atm(self):
        """Capped strangle gamma should be positive near ATM (convex payoff region)."""
        spec = _payoff_spec(_capped_strangle)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        gamma = ov.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert gamma > 0

    def test_capped_strangle_vega_positive(self):
        """Capped strangle vega should be positive (long vol exposure)."""
        spec = _payoff_spec(_capped_strangle)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        vega = ov.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert vega > 0

    def test_theta_negative(self):
        """Custom payoff theta should be negative (time decay)."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        theta = ov.theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert theta < 0

    def test_rho_finite(self):
        """Custom payoff rho should be finite."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=300)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)
        rho = ov.rho(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isfinite(rho)

    def test_tree_delta_matches_numerical_delta(self):
        """Binomial tree delta should be close to numerical delta for PayoffSpec."""
        spec = _payoff_spec(_bull_call_spread)
        params = BinomialParams(num_steps=500)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)

        delta_tree = ov.delta(greek_calc_method=GreekCalculationMethod.TREE)
        delta_num = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(delta_tree, delta_num, rtol=0.05), (
            f"tree={delta_tree:.6f} vs numerical={delta_num:.6f}"
        )

    def test_tree_gamma_matches_numerical_gamma(self):
        """Binomial tree gamma should be close to numerical gamma for capped strangle."""
        spec = _payoff_spec(_capped_strangle)
        params = BinomialParams(num_steps=500)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)

        gamma_tree = ov.gamma(greek_calc_method=GreekCalculationMethod.TREE)
        gamma_num = ov.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        # Capped strangle has meaningful gamma near ATM
        assert np.isclose(gamma_tree, gamma_num, rtol=0.10), (
            f"tree={gamma_tree:.6f} vs numerical={gamma_num:.6f}"
        )

    def test_tree_theta_matches_numerical_theta(self):
        """Binomial tree theta should be close to numerical theta for PayoffSpec."""
        spec = _payoff_spec(_capped_strangle)
        params = BinomialParams(num_steps=500)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=params)

        theta_tree = ov.theta(greek_calc_method=GreekCalculationMethod.TREE)
        theta_num = ov.theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert theta_tree < 0
        assert theta_num < 0
        assert np.isclose(theta_tree, theta_num, rtol=0.10), (
            f"tree={theta_tree:.6f} vs numerical={theta_num:.6f}"
        )

    def test_mc_numerical_delta_close_to_binomial(self):
        """MC numerical delta should be close to binomial for European PayoffSpec."""
        spec = _payoff_spec(_bull_call_spread)

        binom_params = BinomialParams(num_steps=500)
        delta_binom = OptionValuation(
            _ud(), spec, PricingMethod.BINOMIAL, params=binom_params
        ).delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        mc_params = MonteCarloParams(random_seed=MC_SEED)
        delta_mc = OptionValuation(_gbm(), spec, PricingMethod.MONTE_CARLO, params=mc_params).delta(
            greek_calc_method=GreekCalculationMethod.NUMERICAL
        )

        assert np.isclose(delta_binom, delta_mc, rtol=0.05), (
            f"binomial={delta_binom:.6f} vs MC={delta_mc:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Rejection — BSM only
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecRejection:
    """BSM should reject PayoffSpec."""

    def test_bsm_rejects_payoff_spec(self):
        """BSM requires CALL/PUT option_type, unavailable for PayoffSpec."""
        spec = _payoff_spec(_bull_call_spread)
        with pytest.raises(UnsupportedFeatureError, match="BSM"):
            OptionValuation(_ud(), spec, PricingMethod.BSM)

    def test_mc_pathwise_rejects_payoff_spec(self):
        """MC pathwise/LR greeks require VanillaSpec."""
        spec = _payoff_spec(_bull_call_spread)
        mc_params = MonteCarloParams(random_seed=MC_SEED)
        gbm = _gbm()
        ov = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=mc_params)

        with pytest.raises(ValidationError, match="vanilla European options"):
            ov.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)

        with pytest.raises(ValidationError, match="vanilla European options"):
            ov.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)

        # Numerical Greeks remain available for PayoffSpec.
        delta_num = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isfinite(delta_num)


# ═══════════════════════════════════════════════════════════════════════════
# Bull spread vs replicating vanilla calls
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecReplication:
    """Verify PayoffSpec PV matches replication via vanilla calls."""

    def test_bull_spread_matches_vanilla_replication(self):
        """PayoffSpec bull spread should match long 95C - short 115C."""
        ud = _ud()
        params = BinomialParams(num_steps=500)

        # PayoffSpec
        payoff_pv = OptionValuation(
            ud, _payoff_spec(_bull_call_spread), PricingMethod.BINOMIAL, params=params
        ).present_value()

        # Replication via vanilla
        long_95c = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=95.0,
            maturity=MATURITY,
            currency=CURRENCY,
        )
        short_115c = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=115.0,
            maturity=MATURITY,
            currency=CURRENCY,
        )
        long_pv = OptionValuation(
            ud, long_95c, PricingMethod.BINOMIAL, params=params
        ).present_value()
        short_pv = OptionValuation(
            ud, short_115c, PricingMethod.BINOMIAL, params=params
        ).present_value()
        replication_pv = long_pv - short_pv

        assert np.isclose(payoff_pv, replication_pv, rtol=0.005), (
            f"PayoffSpec={payoff_pv:.4f} vs replication={replication_pv:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PDE pricing for custom payoffs
# ═══════════════════════════════════════════════════════════════════════════

# Reference PDEParams with enough resolution for good accuracy.
_PDE_PARAMS = PDEParams(spot_steps=400, time_steps=400)
_PDE_PARAMS_AM = PDEParams(spot_steps=400, time_steps=400, omega=1.2, tol=1e-8, max_iter=2000)
_BINOM_PARAMS = BinomialParams(num_steps=500)


class TestPayoffSpecPDE:
    """PDE finite-difference pricing with affine boundary-model conditions."""

    @pytest.mark.parametrize(
        "payoff_fn, label, exercise",
        [
            (_bull_call_spread, "bull_spread", ExerciseType.EUROPEAN),
            (_bull_call_spread, "bull_spread", ExerciseType.AMERICAN),
            (_capped_strangle, "capped_strangle", ExerciseType.EUROPEAN),
            (_capped_strangle, "capped_strangle", ExerciseType.AMERICAN),
        ],
        ids=[
            "bull_spread-european",
            "bull_spread-american",
            "capped_strangle-european",
            "capped_strangle-american",
        ],
    )
    def test_pde_matches_binomial(self, payoff_fn, label, exercise):
        """PDE should agree with a high-step binomial for various payoffs."""
        ud = _ud()
        spec = _payoff_spec(payoff_fn, exercise_type=exercise)
        is_american = exercise is ExerciseType.AMERICAN

        pde_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS_AM if is_american else _PDE_PARAMS,
        ).present_value()
        binom_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).present_value()

        assert np.isclose(pde_pv, binom_pv, rtol=0.01), (
            f"PDE={pde_pv:.4f} vs binomial={binom_pv:.4f} [{label}, {exercise.value}]"
        )

    def test_pde_digital_call_matches_binomial(self):
        """Digital call: higher-resolution PDE/binomial runs should agree tightly."""
        ud = _ud()
        spec = _payoff_spec(_digital_call)
        pde_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=400, time_steps=400),
        ).present_value()
        binom_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=2000),
        ).present_value()

        assert np.isclose(pde_pv, binom_pv, rtol=0.02), (
            f"PDE={pde_pv:.4f} vs binomial={binom_pv:.4f}"
        )

    def test_pde_digital_call_bounded(self):
        """Digital call: PDE should produce a positive value below the payout."""
        ud = _ud()
        spec = _payoff_spec(_digital_call)
        pv = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=_PDE_PARAMS).present_value()

        assert pv > 0.0
        # Digital pays 1 max, so PV ≤ exp(-rT) * 1
        assert pv < 1.0

    def test_pde_bull_spread_explicit_boundary_model(self):
        """Explicit boundary model on PayoffSpec should give same result as fitted."""
        ud = _ud()

        # Inferred
        spec_inferred = _payoff_spec(_bull_call_spread)
        pv_inferred = OptionValuation(
            ud, spec_inferred, PricingMethod.PDE_FD, params=_PDE_PARAMS
        ).present_value()

        # Explicit: left (0, 0), right (0, 20) since spread caps at 115-95=20
        spec_explicit = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=MATURITY,
            payoff_fn=_bull_call_spread,
            currency=CURRENCY,
            boundary_model=PayoffBoundaryModel(
                left=WingBoundary(slope=0.0, intercept=0.0),
                right=WingBoundary(slope=0.0, intercept=20.0),
            ),
        )
        pv_explicit = OptionValuation(
            ud, spec_explicit, PricingMethod.PDE_FD, params=_PDE_PARAMS
        ).present_value()

        assert np.isclose(pv_inferred, pv_explicit, rtol=1e-4), (
            f"inferred={pv_inferred:.6f} vs explicit={pv_explicit:.6f}"
        )

    def test_pde_bull_spread_replication_vs_vanilla_pde(self):
        """PDE PayoffSpec bull spread should match sum of vanilla PDE calls."""
        ud = _ud()
        pde_params = _PDE_PARAMS

        spec_pv = OptionValuation(
            ud, _payoff_spec(_bull_call_spread), PricingMethod.PDE_FD, params=pde_params
        ).present_value()

        long_95c = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=95.0,
            maturity=MATURITY,
            currency=CURRENCY,
        )
        short_115c = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=115.0,
            maturity=MATURITY,
            currency=CURRENCY,
        )
        replication_pv = (
            OptionValuation(ud, long_95c, PricingMethod.PDE_FD, params=pde_params).present_value()
            - OptionValuation(
                ud, short_115c, PricingMethod.PDE_FD, params=pde_params
            ).present_value()
        )

        assert np.isclose(spec_pv, replication_pv, rtol=0.01), (
            f"PayoffSpec PDE={spec_pv:.4f} vs vanilla replication PDE={replication_pv:.4f}"
        )

    def test_pde_american_ge_european(self):
        """American PDE price should be >= European PDE price."""
        ud = _ud()
        spec_eu = _payoff_spec(_bull_call_spread, exercise_type=ExerciseType.EUROPEAN)
        spec_am = _payoff_spec(_bull_call_spread, exercise_type=ExerciseType.AMERICAN)

        pv_eu = OptionValuation(
            ud, spec_eu, PricingMethod.PDE_FD, params=_PDE_PARAMS
        ).present_value()
        pv_am = OptionValuation(
            ud, spec_am, PricingMethod.PDE_FD, params=_PDE_PARAMS_AM
        ).present_value()

        assert pv_am >= pv_eu - 1e-8, f"American={pv_am:.4f} < European={pv_eu:.4f}"
