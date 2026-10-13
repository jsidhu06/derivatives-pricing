"""Tests for pathwise and likelihood-ratio MC Greeks (European only)."""

import datetime as dt
import logging

import numpy as np
import pytest

from derivatives_pricing.enums import (
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.exceptions import UnsupportedFeatureError, ValidationError
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)

from derivatives_pricing.valuation import (
    MonteCarloParams,
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)

_MISSING: object = object()  # sentinel to distinguish "not passed" from None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MCGreekTestBase:
    """Shared setup for MC Greek tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.spot = 100.0
        self.strike = 100.0
        self.rate = 0.05
        self.vol = 0.20
        self.div_yield = 0.02
        self.currency = "USD"
        self.n_paths = 500_000
        self.seed = 42

        self.csr = DiscountCurve.flat(self.rate)
        self.qcr = DiscountCurve.flat(self.div_yield)
        self.market_data = MarketData(self.pricing_date, self.csr, currency=self.currency)

    # ---------- factories ----------

    def _make_process(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        q_curve: DiscountCurve | None | object = _MISSING,
    ) -> GBMProcess:
        if q_curve is _MISSING:
            resolved_q_curve: DiscountCurve = self.qcr
        elif q_curve is None or isinstance(q_curve, DiscountCurve):
            resolved_q_curve = q_curve
        else:
            raise TypeError("q_curve must be DiscountCurve, None, or _MISSING")
        params = GBMParams(
            initial_value=spot or self.spot,
            volatility=vol or self.vol,
            dividend_curve=resolved_q_curve,
        )
        sim_config = SimulationConfig(
            paths=self.n_paths,
            num_steps=12,
            end_date=self.maturity,
        )
        return GBMProcess(self.market_data, params, sim_config)

    def _make_val(
        self,
        option_type: OptionType,
        *,
        strike: float | None = None,
        process: GBMProcess | None = None,
    ) -> OptionValuation:
        spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike or self.strike,
            maturity=self.maturity,
        )
        mc_params = MonteCarloParams(random_seed=self.seed)
        return OptionValuation(
            underlying=process or self._make_process(),
            spec=spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=mc_params,
        )

    def _bsm_ref(
        self,
        option_type: OptionType,
        *,
        strike: float | None = None,
    ) -> OptionValuation:
        """Build a BSM valuation to compare against."""
        ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.vol,
            market_data=self.market_data,
            dividend_curve=self.qcr,
        )
        spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike or self.strike,
            maturity=self.maturity,
        )
        return OptionValuation(ud, spec, PricingMethod.ANALYTICAL)


# ---------------------------------------------------------------------------
# Pathwise delta
# ---------------------------------------------------------------------------


class TestPathwiseDelta(_MCGreekTestBase):
    """Pathwise (IPA) delta vs BSM analytical for calls and puts."""

    def test_call_atm(self):
        mc_val = self._make_val(OptionType.CALL)
        bsm_delta = self._bsm_ref(OptionType.CALL).delta()
        pw_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_delta, bsm_delta, rtol=0.01), (
            f"pathwise call delta {pw_delta:.6f} vs BSM {bsm_delta:.6f}"
        )

    def test_put_atm(self):
        mc_val = self._make_val(OptionType.PUT)
        bsm_delta = self._bsm_ref(OptionType.PUT).delta()
        pw_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_delta, bsm_delta, rtol=0.01), (
            f"pathwise put delta {pw_delta:.6f} vs BSM {bsm_delta:.6f}"
        )

    def test_call_itm(self):
        mc_val = self._make_val(OptionType.CALL, strike=80.0)
        bsm_delta = self._bsm_ref(OptionType.CALL, strike=80.0).delta()
        pw_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_delta, bsm_delta, rtol=0.01)

    def test_put_otm(self):
        mc_val = self._make_val(OptionType.PUT, strike=80.0)
        bsm_delta = self._bsm_ref(OptionType.PUT, strike=80.0).delta()
        pw_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_delta, bsm_delta, rtol=0.01)

    def test_call_delta_positive(self):
        d = self._make_val(OptionType.CALL).delta(
            greek_calc_method=GreekCalculationMethod.PATHWISE,
        )
        assert d > 0

    def test_put_delta_negative(self):
        d = self._make_val(OptionType.PUT).delta(
            greek_calc_method=GreekCalculationMethod.PATHWISE,
        )
        assert d < 0

    def test_call_delta_bounded(self):
        d = self._make_val(OptionType.CALL).delta(
            greek_calc_method=GreekCalculationMethod.PATHWISE,
        )
        assert 0 < d < 1.0

    def test_put_delta_bounded(self):
        d = self._make_val(OptionType.PUT).delta(
            greek_calc_method=GreekCalculationMethod.PATHWISE,
        )
        assert -1.0 < d < 0


# ---------------------------------------------------------------------------
# Likelihood-ratio delta
# ---------------------------------------------------------------------------


class TestLRDelta(_MCGreekTestBase):
    """Likelihood-ratio delta vs BSM analytical."""

    def test_call_atm(self):
        mc_val = self._make_val(OptionType.CALL)
        bsm_delta = self._bsm_ref(OptionType.CALL).delta()
        lr_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        assert np.isclose(lr_delta, bsm_delta, rtol=0.01), (
            f"LR call delta {lr_delta:.6f} vs BSM {bsm_delta:.6f}"
        )

    def test_put_atm(self):
        mc_val = self._make_val(OptionType.PUT)
        bsm_delta = self._bsm_ref(OptionType.PUT).delta()
        lr_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        assert np.isclose(lr_delta, bsm_delta, rtol=0.01), (
            f"LR put delta {lr_delta:.6f} vs BSM {bsm_delta:.6f}"
        )

    def test_call_otm(self):
        mc_val = self._make_val(OptionType.CALL, strike=120.0)
        bsm_delta = self._bsm_ref(OptionType.CALL, strike=120.0).delta()
        lr_delta = mc_val.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        assert np.isclose(lr_delta, bsm_delta, rtol=0.01)


# ---------------------------------------------------------------------------
# Pathwise vega
# ---------------------------------------------------------------------------


class TestPathwiseVega(_MCGreekTestBase):
    """Pathwise vega vs BSM analytical."""

    def test_call_atm(self):
        mc_val = self._make_val(OptionType.CALL)
        bsm_vega = self._bsm_ref(OptionType.CALL).vega()
        pw_vega = mc_val.vega(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_vega, bsm_vega, rtol=0.01), (
            f"pathwise call vega {pw_vega:.6f} vs BSM {bsm_vega:.6f}"
        )

    def test_put_atm(self):
        mc_val = self._make_val(OptionType.PUT)
        bsm_vega = self._bsm_ref(OptionType.PUT).vega()
        pw_vega = mc_val.vega(greek_calc_method=GreekCalculationMethod.PATHWISE)
        assert np.isclose(pw_vega, bsm_vega, rtol=0.01), (
            f"pathwise put vega {pw_vega:.6f} vs BSM {bsm_vega:.6f}"
        )

    def test_vega_always_positive(self):
        for ot in (OptionType.CALL, OptionType.PUT):
            v = self._make_val(ot).vega(
                greek_calc_method=GreekCalculationMethod.PATHWISE,
            )
            assert v > 0, f"{ot.value} pathwise vega should be positive"


# ---------------------------------------------------------------------------
# Likelihood-ratio vega
# ---------------------------------------------------------------------------


class TestLRVega(_MCGreekTestBase):
    """Likelihood-ratio vega vs BSM analytical."""

    def test_call_atm(self):
        mc_val = self._make_val(OptionType.CALL)
        bsm_vega = self._bsm_ref(OptionType.CALL).vega()
        lr_vega = mc_val.vega(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        assert np.isclose(lr_vega, bsm_vega, rtol=0.01), (
            f"LR call vega {lr_vega:.6f} vs BSM {bsm_vega:.6f}"
        )

    def test_put_atm(self):
        mc_val = self._make_val(OptionType.PUT)
        bsm_vega = self._bsm_ref(OptionType.PUT).vega()
        lr_vega = mc_val.vega(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        assert np.isclose(lr_vega, bsm_vega, rtol=0.01), (
            f"LR put vega {lr_vega:.6f} vs BSM {bsm_vega:.6f}"
        )


# ---------------------------------------------------------------------------
# Cross-method consistency (pathwise ≈ LR ≈ numerical)
# ---------------------------------------------------------------------------


class TestMCGreekCrossConsistency(_MCGreekTestBase):
    """All three MC Greek methods should agree within MC noise."""

    def test_delta_three_methods_agree(self):
        mc_val = self._make_val(OptionType.CALL)
        d_pw = mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        d_lr = mc_val.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        d_num = mc_val.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(d_pw, d_lr, rtol=0.01), f"pathwise {d_pw:.6f} vs LR {d_lr:.6f}"
        assert np.isclose(d_pw, d_num, rtol=0.01), f"pathwise {d_pw:.6f} vs num {d_num:.6f}"

    def test_vega_three_methods_agree(self):
        mc_val = self._make_val(OptionType.CALL)
        v_pw = mc_val.vega(greek_calc_method=GreekCalculationMethod.PATHWISE)
        v_lr = mc_val.vega(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        v_num = mc_val.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isclose(v_pw, v_lr, rtol=0.01), f"pathwise {v_pw:.6f} vs LR {v_lr:.6f}"
        assert np.isclose(v_pw, v_num, rtol=0.01), f"pathwise {v_pw:.6f} vs num {v_num:.6f}"


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestMCGreekValidation(_MCGreekTestBase):
    """Ensure PATHWISE/LR rejected for wrong pricing methods / greeks."""

    def test_pathwise_rejects_bsm(self):
        bsm = self._bsm_ref(OptionType.CALL)
        with pytest.raises(ValidationError, match="pathwise"):
            bsm.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)

    def test_lr_rejects_bsm(self):
        bsm = self._bsm_ref(OptionType.CALL)
        with pytest.raises(ValidationError, match="likelihood_ratio"):
            bsm.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)

    def test_lr_rejects_gamma(self):
        mc_val = self._make_val(OptionType.CALL)
        with pytest.raises(ValidationError, match="likelihood_ratio"):
            mc_val.gamma(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)

    def test_pathwise_rejects_discrete_dividends(self):
        params = GBMParams(
            initial_value=100.0,
            volatility=0.20,
            discrete_dividends=[(dt.datetime(2025, 6, 15), 2.0)],
        )
        sim_config = SimulationConfig(
            paths=10_000,
            num_steps=12,
            end_date=dt.datetime(2026, 1, 1),
        )
        process = GBMProcess(self.market_data, params, sim_config)
        mc_val = self._make_val(OptionType.CALL, process=process)
        with pytest.raises(UnsupportedFeatureError, match="discrete dividends"):
            mc_val.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)

    def test_lr_rejects_discrete_dividends(self):
        params = GBMParams(
            initial_value=100.0,
            volatility=0.20,
            discrete_dividends=[(dt.datetime(2025, 6, 15), 2.0)],
        )
        sim_config = SimulationConfig(
            paths=10_000,
            num_steps=12,
            end_date=dt.datetime(2026, 1, 1),
        )
        process = GBMProcess(self.market_data, params, sim_config)
        mc_val = self._make_val(OptionType.CALL, process=process)
        with pytest.raises(UnsupportedFeatureError, match="discrete dividends"):
            mc_val.delta(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)


# ---------------------------------------------------------------------------
# No-dividend edge case
# ---------------------------------------------------------------------------


class TestMCGreekNoDividend(_MCGreekTestBase):
    """Verify MC Greeks work when there is no dividend yield."""

    def test_pathwise_delta_no_div(self):
        process = self._make_process(q_curve=None)
        spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
        )
        mc = OptionValuation(
            process,
            spec,
            PricingMethod.MONTE_CARLO,
            MonteCarloParams(random_seed=self.seed),
        )
        bsm_ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.vol,
            market_data=self.market_data,
        )
        bsm = OptionValuation(bsm_ud, spec, PricingMethod.ANALYTICAL)
        pw_delta = mc.delta(greek_calc_method=GreekCalculationMethod.PATHWISE)
        bsm_delta = bsm.delta()
        assert np.isclose(pw_delta, bsm_delta, rtol=0.01)


# ---------------------------------------------------------------------------
# Parameterised cross-method agreement across market scenarios
# ---------------------------------------------------------------------------

# Each scenario: (label, option_type, strike, vol, rate_curve_spec, div_curve_spec)
# Curve specs: "flat" uses the base-class defaults; a dict is passed to DiscountCurve.from_forwards.

_SCENARIOS = [
    pytest.param(OptionType.CALL, 120.0, 0.20, "flat", "flat", id="otm_call"),
    pytest.param(OptionType.PUT, 120.0, 0.25, "flat", "flat", id="itm_put_high_vol"),
    pytest.param(OptionType.CALL, 100.0, 0.20, "flat", None, id="atm_no_div"),
    pytest.param(
        OptionType.CALL,
        110.0,
        0.25,
        {"times": np.array([0.0, 0.5, 1.0]), "forwards": np.array([0.04, 0.06])},
        {"times": np.array([0.0, 0.5, 1.0]), "forwards": np.array([0.01, 0.03])},
        id="non_flat_both_curves",
    ),
]

_GREEKS = [
    pytest.param("delta", id="delta"),
    pytest.param("gamma", id="gamma"),
    pytest.param("vega", id="vega"),
    pytest.param("theta", id="theta"),
    pytest.param("rho", id="rho"),
]


@pytest.mark.slow
class TestMCGreekMethodAgreement:
    """PATHWISE, LR, and NUMERICAL should agree across diverse market setups."""

    PRICING_DATE = dt.datetime(2025, 1, 1)
    MATURITY = dt.datetime(2026, 1, 1)
    SPOT = 100.0
    N_PATHS = 500_000
    SEED = 42
    CURRENCY = "USD"

    # --- helpers ---

    def _build_curve(self, spec, default_rate: float) -> DiscountCurve | None:
        if spec is None:
            return None
        if spec == "flat":
            return DiscountCurve.flat(default_rate)
        # dict with from_forwards kwargs
        return DiscountCurve.from_forwards(**spec)

    def _build_mc(
        self,
        option_type: OptionType,
        strike: float,
        vol: float,
        rate_spec,
        div_spec,
    ) -> OptionValuation:
        rate_curve = self._build_curve(rate_spec, default_rate=0.05)
        div_curve = self._build_curve(div_spec, default_rate=0.02)
        md = MarketData(self.PRICING_DATE, rate_curve, currency=self.CURRENCY)
        params = GBMParams(initial_value=self.SPOT, volatility=vol, dividend_curve=div_curve)
        sim_config = SimulationConfig(
            paths=self.N_PATHS,
            num_steps=12,
            end_date=self.MATURITY,
        )
        process = GBMProcess(md, params, sim_config)
        spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=self.MATURITY,
        )
        return OptionValuation(
            process,
            spec,
            PricingMethod.MONTE_CARLO,
            MonteCarloParams(random_seed=self.SEED),
        )

    # --- parameterised tests ---

    @pytest.mark.parametrize("greek", _GREEKS)
    @pytest.mark.parametrize("option_type,strike,vol,rate_spec,div_spec", _SCENARIOS)
    def test_three_methods_agree(
        self,
        option_type,
        strike,
        vol,
        rate_spec,
        div_spec,
        greek,
    ):
        mc_val = self._build_mc(option_type, strike, vol, rate_spec, div_spec)
        greek_fn = getattr(mc_val, greek)
        pw = greek_fn(greek_calc_method=GreekCalculationMethod.PATHWISE)
        num = greek_fn(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        lr = None
        if greek != "gamma":
            lr = greek_fn(greek_calc_method=GreekCalculationMethod.LIKELIHOOD_RATIO)
        logger.info(
            "%s %s K=%.0f vol=%.2f | PW=%.8f NUM=%.8f LR=%s",
            greek,
            option_type.value,
            strike,
            vol,
            pw,
            num,
            f"{lr:.8f}" if lr is not None else "N/A",
        )
        assert np.isclose(pw, num, rtol=0.01, atol=0.005), f"{greek} PW {pw:.6f} vs NUM {num:.6f}"
        if lr is not None:
            assert np.isclose(pw, lr, rtol=0.01, atol=0.005), f"{greek} PW {pw:.6f} vs LR {lr:.6f}"


# ---------------------------------------------------------------------------
# Bias bounded by the sampling error
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_pathwise_and_lr_greeks_are_unbiased_against_bsm(option_type):
    """Tolerances come from the sampling error, not a fixed percentage: each
    estimator is averaged over 32 independent seeds and the mean must sit
    within 4 standard errors (seed-to-seed sd / sqrt(32)) of the closed form.
    The bound is 0.2-1% of the Greek for delta, rho and pathwise vega/theta,
    so a 1% bias fails there; the noisier pathwise gamma and LR vega/theta
    get 2-3.5%."""
    pricing_date, maturity = dt.datetime(2025, 1, 1), dt.datetime(2026, 1, 1)
    market_data = MarketData(pricing_date, DiscountCurve.flat(0.05), currency="USD")
    spec = VanillaSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=maturity,
    )
    bsm = OptionValuation(
        UnderlyingData(initial_value=100.0, volatility=0.20, market_data=market_data),
        spec,
        PricingMethod.ANALYTICAL,
    )
    process = GBMProcess(
        market_data,
        GBMParams(initial_value=100.0, volatility=0.20),
        SimulationConfig(paths=20_000, num_steps=26, end_date=maturity),
    )
    pathwise, lr = GreekCalculationMethod.PATHWISE, GreekCalculationMethod.LIKELIHOOD_RATIO
    cases = [
        ("delta", pathwise),
        ("gamma", pathwise),
        ("vega", pathwise),
        ("theta", pathwise),
        ("rho", pathwise),
        ("delta", lr),
        ("vega", lr),
        ("theta", lr),
        ("rho", lr),
    ]
    num_seeds = 32
    samples = np.empty((num_seeds, len(cases)))
    for i, seed in enumerate(range(1, num_seeds + 1)):
        mc = OptionValuation(
            process, spec, PricingMethod.MONTE_CARLO, MonteCarloParams(random_seed=seed)
        )
        samples[i] = [getattr(mc, greek)(greek_calc_method=method) for greek, method in cases]

    means = samples.mean(axis=0)
    se_of_mean = samples.std(axis=0, ddof=1) / np.sqrt(num_seeds)
    for (greek, method), mean, se in zip(cases, means, se_of_mean, strict=True):
        reference = getattr(bsm, greek)()
        assert abs(mean - reference) < 4.0 * se, (method.value, greek, mean, reference, se)
