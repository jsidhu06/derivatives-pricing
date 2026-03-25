"""Integration-style tests for ``PayoffSpec`` custom payoff contracts.

Scope:
- Present value smoke tests and no-arbitrage bounds
- Cross-engine convergence (binomial, MC, PDE) for European and American
- American-vs-European monotonicity across all engines
- Greek behavior:
    - Sign and bound checks
    - Binomial TREE extraction vs numerical bump-and-revalue
    - Cross-engine Greek convergence (binomial, PDE, MC)
- European replication: PayoffSpec vs sum of vanilla legs (binomial and PDE)
- PDE-specific: affine boundary models, digital call bounds, discrete dividends
- Engine guardrails: BSM rejects PayoffSpec, MC pathwise/LR rejected
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


def _bear_put_spread(S):
    """Bear put spread: long 105 put, short 85 put. Max payoff 20."""
    return np.maximum(105.0 - S, 0) - np.maximum(85.0 - S, 0)


def _straddle(S):
    """Long 100 straddle: long 100C + long 100P."""
    return np.maximum(S - 100.0, 0) + np.maximum(100.0 - S, 0)


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


# ---------------------------------------------------------------------------
# Shared engine params
# ---------------------------------------------------------------------------

_BINOM_PARAMS = BinomialParams(num_steps=500)
_PDE_PARAMS = PDEParams(spot_steps=400, time_steps=400)
_PDE_PARAMS_AM = PDEParams(spot_steps=400, time_steps=400, omega=1.2, tol=1e-8, max_iter=2000)
_MC_PARAMS = MonteCarloParams(random_seed=MC_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Present value — smoke tests and bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecPresentValue:
    """Smoke tests: positive PV and no-arbitrage bounds."""

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
        gbm = _gbm()
        ov = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=_MC_PARAMS)
        pv = ov.present_value()
        assert pv > 0, f"{label}: expected positive PV, got {pv}"
        assert np.isfinite(pv)

    def test_bull_spread_bounded_by_max_spread(self):
        """Bull call spread PV must be <= discounted max spread (20)."""
        spec = _payoff_spec(_bull_call_spread)
        pv = OptionValuation(
            _ud(), spec, PricingMethod.BINOMIAL, params=_BINOM_PARAMS
        ).present_value()
        df = np.exp(-RATE * 1.0)
        assert pv <= 20.0 * df + 0.01

    def test_capped_strangle_bounded_by_cap(self):
        """Capped strangle PV must be <= discounted cap (40)."""
        spec = _payoff_spec(_capped_strangle)
        pv = OptionValuation(
            _ud(), spec, PricingMethod.BINOMIAL, params=_BINOM_PARAMS
        ).present_value()
        df = np.exp(-RATE * 1.0)
        assert pv <= 40.0 * df + 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Cross-engine convergence — binomial, MC, PDE
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecCrossEngine:
    """All three engines should produce similar PVs for PayoffSpec."""

    # MC American LS regression on Laguerre basis can struggle with
    # non-monotonic payoffs (capped strangle) — widen tolerance.
    _MC_AM_WIDE = {"capped_strangle", "straddle"}

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_bull_call_spread, "bull_spread"),
            (_bear_put_spread, "bear_spread"),
            (_straddle, "straddle"),
            (_capped_strangle, "capped_strangle"),
        ],
        ids=["bull_spread", "bear_spread", "straddle", "capped_strangle"],
    )
    @pytest.mark.parametrize(
        "exercise",
        [ExerciseType.EUROPEAN, ExerciseType.AMERICAN],
        ids=["european", "american"],
    )
    def test_three_engine_convergence(self, payoff_fn, label, exercise):
        """Binomial, MC, and PDE should agree on PV."""
        ud = _ud()
        is_american = exercise is ExerciseType.AMERICAN
        spec = _payoff_spec(payoff_fn, exercise_type=exercise)

        binom_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).present_value()
        pde_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS_AM if is_american else _PDE_PARAMS,
        ).present_value()

        mc_spec = _payoff_spec(payoff_fn, exercise_type=exercise)
        mc_pv = OptionValuation(
            _gbm(),
            mc_spec,
            PricingMethod.MONTE_CARLO,
            params=_MC_PARAMS,
        ).present_value()

        # PDE vs binomial — tight
        assert np.isclose(pde_pv, binom_pv, rtol=0.02), (
            f"PDE={pde_pv:.4f} vs binom={binom_pv:.4f} [{label}, {exercise.value}]"
        )

        # MC vs binomial — wider for American with non-monotonic payoffs
        # (LS lower-bound bias from Laguerre regression on moneyness)
        mc_rtol = 0.08 if (is_american and label in self._MC_AM_WIDE) else 0.03
        assert np.isclose(mc_pv, binom_pv, rtol=mc_rtol), (
            f"MC={mc_pv:.4f} vs binom={binom_pv:.4f} [{label}, {exercise.value}]"
        )

    def test_digital_call_pde_matches_binomial(self):
        """Digital call: PDE and high-step binomial should agree (European)."""
        ud = _ud()
        spec = _payoff_spec(_digital_call)
        pde_pv = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
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


# ═══════════════════════════════════════════════════════════════════════════
# American >= European monotonicity
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecAmerican:
    """American PayoffSpec should produce >= European PV across all engines."""

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_capped_strangle, "capped_strangle"),
            (_bull_call_spread, "bull_spread"),
        ],
        ids=["capped_strangle", "bull_spread"],
    )
    @pytest.mark.parametrize(
        "engine, params_eu, params_am, ud_or_gbm",
        [
            ("binomial", _BINOM_PARAMS, _BINOM_PARAMS, "ud"),
            ("pde", _PDE_PARAMS, _PDE_PARAMS_AM, "ud"),
            ("mc", _MC_PARAMS, _MC_PARAMS, "gbm"),
        ],
        ids=["binomial", "pde", "mc"],
    )
    def test_american_ge_european(self, payoff_fn, label, engine, params_eu, params_am, ud_or_gbm):
        method = (
            PricingMethod.MONTE_CARLO
            if engine == "mc"
            else (PricingMethod.PDE_FD if engine == "pde" else PricingMethod.BINOMIAL)
        )
        underlying = _gbm() if ud_or_gbm == "gbm" else _ud()

        euro_spec = _payoff_spec(payoff_fn, ExerciseType.EUROPEAN)
        amer_spec = _payoff_spec(payoff_fn, ExerciseType.AMERICAN)

        euro_pv = OptionValuation(
            underlying,
            euro_spec,
            method,
            params=params_eu,
        ).present_value()
        amer_pv = OptionValuation(
            underlying,
            amer_spec,
            method,
            params=params_am,
        ).present_value()

        assert amer_pv >= euro_pv - 1e-8, (
            f"{label} [{engine}]: American={amer_pv:.4f} < European={euro_pv:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Greeks — sign/bound checks and tree vs numerical
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecGreeks:
    """Greek sign/bound checks and binomial tree vs numerical consistency."""

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
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=_BINOM_PARAMS)

        delta_tree = ov.delta(greek_calc_method=GreekCalculationMethod.TREE)
        delta_num = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(delta_tree, delta_num, rtol=0.05), (
            f"tree={delta_tree:.6f} vs numerical={delta_num:.6f}"
        )

    def test_tree_gamma_matches_numerical_gamma(self):
        """Binomial tree gamma should be close to numerical gamma for capped strangle."""
        spec = _payoff_spec(_capped_strangle)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=_BINOM_PARAMS)

        gamma_tree = ov.gamma(greek_calc_method=GreekCalculationMethod.TREE)
        gamma_num = ov.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(gamma_tree, gamma_num, rtol=0.10), (
            f"tree={gamma_tree:.6f} vs numerical={gamma_num:.6f}"
        )

    def test_tree_theta_matches_numerical_theta(self):
        """Binomial tree theta should be close to numerical theta for PayoffSpec."""
        spec = _payoff_spec(_capped_strangle)
        ov = OptionValuation(_ud(), spec, PricingMethod.BINOMIAL, params=_BINOM_PARAMS)

        theta_tree = ov.theta(greek_calc_method=GreekCalculationMethod.TREE)
        theta_num = ov.theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert theta_tree < 0
        assert theta_num < 0
        assert np.isclose(theta_tree, theta_num, rtol=0.10), (
            f"tree={theta_tree:.6f} vs numerical={theta_num:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Greeks — cross-engine convergence
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecGreeksCrossEngine:
    """Bump-and-revalue Greeks should agree across binomial, PDE, and MC."""

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_bull_call_spread, "bull_spread"),
            (_straddle, "straddle"),
            (_capped_strangle, "capped_strangle"),
        ],
        ids=["bull_spread", "straddle", "capped_strangle"],
    )
    def test_delta_european(self, payoff_fn, label):
        """European delta: binomial, PDE, MC should agree."""
        ud = _ud()
        spec = _payoff_spec(payoff_fn)

        d_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).delta()
        d_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).delta()
        d_mc = OptionValuation(
            _gbm(),
            _payoff_spec(payoff_fn),
            PricingMethod.MONTE_CARLO,
            params=_MC_PARAMS,
        ).delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(d_pde, d_binom, rtol=0.05), (
            f"delta PDE={d_pde:.6f} vs binom={d_binom:.6f} [{label}]"
        )
        assert np.isclose(d_mc, d_binom, rtol=0.05), (
            f"delta MC={d_mc:.6f} vs binom={d_binom:.6f} [{label}]"
        )

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_bull_call_spread, "bull_spread"),
            (_straddle, "straddle"),
        ],
        ids=["bull_spread", "straddle"],
    )
    def test_delta_american_pde_vs_binomial(self, payoff_fn, label):
        """American delta: PDE should agree with binomial."""
        ud = _ud()
        spec = _payoff_spec(payoff_fn, exercise_type=ExerciseType.AMERICAN)

        d_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).delta()
        d_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS_AM,
        ).delta()

        assert np.isclose(d_pde, d_binom, rtol=0.05), (
            f"delta PDE={d_pde:.6f} vs binom={d_binom:.6f} [{label}, american]"
        )

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_bull_call_spread, "bull_spread"),
            (_straddle, "straddle"),
            (_capped_strangle, "capped_strangle"),
        ],
        ids=["bull_spread", "straddle", "capped_strangle"],
    )
    def test_vega_european(self, payoff_fn, label):
        """European vega: PDE and binomial should agree."""
        ud = _ud()
        spec = _payoff_spec(payoff_fn)

        v_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).vega()
        v_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).vega()

        assert np.isclose(v_pde, v_binom, rtol=0.10), (
            f"vega PDE={v_pde:.6f} vs binom={v_binom:.6f} [{label}]"
        )

    @pytest.mark.parametrize(
        "payoff_fn, label",
        [
            (_bull_call_spread, "bull_spread"),
            (_straddle, "straddle"),
        ],
        ids=["bull_spread", "straddle"],
    )
    def test_theta_european(self, payoff_fn, label):
        """European theta: PDE and binomial should agree."""
        ud = _ud()
        spec = _payoff_spec(payoff_fn)

        t_pde = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).theta()
        t_binom = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=_BINOM_PARAMS,
        ).theta()

        assert np.isclose(t_pde, t_binom, rtol=0.10, atol=0.01), (
            f"theta PDE={t_pde:.6f} vs binom={t_binom:.6f} [{label}]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Replication — PayoffSpec vs sum of vanilla legs (European only)
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecReplication:
    """PayoffSpec PV should match sum of vanilla legs for European exercise."""

    def _vanilla(self, option_type, strike):
        return VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=MATURITY,
            currency=CURRENCY,
        )

    @pytest.mark.parametrize(
        "engine, params",
        [
            (PricingMethod.BINOMIAL, _BINOM_PARAMS),
            (PricingMethod.PDE_FD, _PDE_PARAMS),
        ],
        ids=["binomial", "pde"],
    )
    def test_bull_spread_replication(self, engine, params):
        """Bull spread PayoffSpec should match long 95C - short 115C."""
        ud = _ud()
        spec_pv = OptionValuation(
            ud,
            _payoff_spec(_bull_call_spread),
            engine,
            params=params,
        ).present_value()

        replication_pv = (
            OptionValuation(
                ud, self._vanilla(OptionType.CALL, 95.0), engine, params=params
            ).present_value()
            - OptionValuation(
                ud, self._vanilla(OptionType.CALL, 115.0), engine, params=params
            ).present_value()
        )

        assert np.isclose(spec_pv, replication_pv, rtol=0.01), (
            f"PayoffSpec={spec_pv:.4f} vs replication={replication_pv:.4f} [{engine.value}]"
        )

    @pytest.mark.parametrize(
        "engine, params",
        [
            (PricingMethod.BINOMIAL, _BINOM_PARAMS),
            (PricingMethod.PDE_FD, _PDE_PARAMS),
        ],
        ids=["binomial", "pde"],
    )
    def test_capped_strangle_replication(self, engine, params):
        """Capped strangle should match long 90P + long 110C, short 50P + short 150C."""
        ud = _ud()
        spec_pv = OptionValuation(
            ud,
            _payoff_spec(_capped_strangle),
            engine,
            params=params,
        ).present_value()

        def _pv(option_type, strike):
            return OptionValuation(
                ud,
                self._vanilla(option_type, strike),
                engine,
                params=params,
            ).present_value()

        replication_pv = (
            _pv(OptionType.PUT, 90.0)  # long 90 put
            - _pv(OptionType.PUT, 50.0)  # short 50 put (caps put wing at 40)
            + _pv(OptionType.CALL, 110.0)  # long 110 call
            - _pv(OptionType.CALL, 150.0)  # short 150 call (caps call wing at 40)
        )

        assert np.isclose(spec_pv, replication_pv, rtol=0.01), (
            f"PayoffSpec={spec_pv:.4f} vs replication={replication_pv:.4f} [{engine.value}]"
        )

    @pytest.mark.parametrize(
        "engine, params",
        [
            (PricingMethod.BINOMIAL, _BINOM_PARAMS),
            (PricingMethod.PDE_FD, _PDE_PARAMS),
        ],
        ids=["binomial", "pde"],
    )
    def test_straddle_replication(self, engine, params):
        """Straddle PayoffSpec should match long 100C + long 100P."""
        ud = _ud()
        spec_pv = OptionValuation(
            ud,
            _payoff_spec(_straddle),
            engine,
            params=params,
        ).present_value()

        replication_pv = (
            OptionValuation(
                ud, self._vanilla(OptionType.CALL, 100.0), engine, params=params
            ).present_value()
            + OptionValuation(
                ud, self._vanilla(OptionType.PUT, 100.0), engine, params=params
            ).present_value()
        )

        assert np.isclose(spec_pv, replication_pv, rtol=0.01), (
            f"PayoffSpec={spec_pv:.4f} vs replication={replication_pv:.4f} [{engine.value}]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PDE-specific — boundary models, digital bounds, discrete dividends
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecPDESpecific:
    """Tests for PDE-only concerns: boundary models, bounds, dividends."""

    def test_digital_call_bounded(self):
        """Digital call: PDE should produce a positive value below the payout."""
        ud = _ud()
        spec = _payoff_spec(_digital_call)
        pv = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=_PDE_PARAMS).present_value()

        assert pv > 0.0
        assert pv < 1.0  # digital pays 1 max, so PV <= exp(-rT)

    def test_explicit_boundary_model_matches_auto_fit(self):
        """Explicit PayoffBoundaryModel should give same result as auto-fitted."""
        ud = _ud()

        spec_inferred = _payoff_spec(_bull_call_spread)
        pv_inferred = OptionValuation(
            ud,
            spec_inferred,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).present_value()

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
            ud,
            spec_explicit,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).present_value()

        assert np.isclose(pv_inferred, pv_explicit, rtol=1e-4), (
            f"inferred={pv_inferred:.6f} vs explicit={pv_explicit:.6f}"
        )

    def test_american_ge_european(self):
        """American PDE price should be >= European PDE price."""
        ud = _ud()
        spec_eu = _payoff_spec(_bull_call_spread, exercise_type=ExerciseType.EUROPEAN)
        spec_am = _payoff_spec(_bull_call_spread, exercise_type=ExerciseType.AMERICAN)

        pv_eu = OptionValuation(
            ud,
            spec_eu,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS,
        ).present_value()
        pv_am = OptionValuation(
            ud,
            spec_am,
            PricingMethod.PDE_FD,
            params=_PDE_PARAMS_AM,
        ).present_value()

        assert pv_am >= pv_eu - 1e-8, f"American={pv_am:.4f} < European={pv_eu:.4f}"

    def test_discrete_dividends(self):
        """PDE with discrete dividends should agree with binomial for PayoffSpec."""
        mid_date = dt.datetime(2025, 7, 1)
        divs = [(mid_date, 2.0)]
        ud = UnderlyingData(
            initial_value=SPOT,
            volatility=VOL,
            market_data=_md(),
            discrete_dividends=divs,
        )

        for payoff_fn, label in [
            (_bull_call_spread, "bull_spread"),
            (_straddle, "straddle"),
        ]:
            for exercise in (ExerciseType.EUROPEAN, ExerciseType.AMERICAN):
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

                assert np.isclose(pde_pv, binom_pv, rtol=0.02), (
                    f"{label} [{exercise.value}] with divs: "
                    f"PDE={pde_pv:.4f} vs binom={binom_pv:.4f}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Rejection — BSM and MC pathwise/LR
# ═══════════════════════════════════════════════════════════════════════════


class TestPayoffSpecRejection:
    """Engine guardrails: BSM rejects PayoffSpec, MC pathwise/LR rejected."""

    def test_bsm_rejects_payoff_spec(self):
        """BSM requires CALL/PUT option_type, unavailable for PayoffSpec."""
        spec = _payoff_spec(_bull_call_spread)
        with pytest.raises(UnsupportedFeatureError, match="BSM"):
            OptionValuation(_ud(), spec, PricingMethod.BSM)

    def test_mc_pathwise_rejects_payoff_spec(self):
        """MC pathwise/LR greeks require VanillaSpec."""
        spec = _payoff_spec(_bull_call_spread)
        gbm = _gbm()
        ov = OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=_MC_PARAMS)

        with pytest.raises(ValidationError, match="vanilla European options"):
            ov.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)

        with pytest.raises(ValidationError, match="vanilla European options"):
            ov.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)

        # Numerical Greeks remain available for PayoffSpec.
        delta_num = ov.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isfinite(delta_num)
