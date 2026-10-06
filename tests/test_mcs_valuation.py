"""Tests for Monte Carlo Simulation valuation."""

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.exceptions import UnsupportedFeatureError
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from conftest import (
    BINOM_PARAMS,
    MC_PARAMS,
    PRICING_DATE,
    MATURITY,
    RATE,
    SPOT,
    VOL,
)
from helpers import CURRENCY, flat_curve, market_data, pv, spec
from derivatives_pricing.valuation import (
    BarrierSpec,
    MonteCarloParams,
    OptionValuation,
    PayoffSpec,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# BSM reference (S=100, K=100, r=0.05, σ=0.20, T=1)
# ---------------------------------------------------------------------------
_BSM_ATM_CALL = 10.4506
_BSM_ATM_PUT = 5.5735
_MC_RTOL_ATM = 0.03
_MC_RTOL_CROSS_METHOD = 0.03


def _market_data(discount_curve: DiscountCurve | None = None):
    curve = (
        discount_curve if discount_curve is not None else flat_curve(PRICING_DATE, MATURITY, RATE)
    )
    return market_data(
        pricing_date=PRICING_DATE,
        discount_curve=curve,
    )


def _gbm(
    paths: int = 10_000,
    num_steps: int = 365,
    spot: float = SPOT,
    vol: float = VOL,
    discount_curve: DiscountCurve | None = None,
) -> GBMProcess:
    params = GBMParams(initial_value=spot, volatility=vol)
    sim_config = SimulationConfig(paths=paths, num_steps=num_steps, end_date=MATURITY)
    return GBMProcess(_market_data(discount_curve), params, sim_config)


class TestMCSValuation:
    """Tests for Monte Carlo Simulation valuation."""

    @pytest.mark.parametrize(
        "option_type,expected",
        [
            (OptionType.CALL, _BSM_ATM_CALL),
            (OptionType.PUT, _BSM_ATM_PUT),
        ],
    )
    def test_mcs_european_atm_matches_bsm_reference(self, option_type: OptionType, expected: float):
        """MCS ATM prices should stay close to BSM references within sampling noise."""
        result = pv(_gbm(), spec(option_type), PricingMethod.MONTE_CARLO, params=MC_PARAMS)
        assert np.isclose(result, expected, rtol=_MC_RTOL_ATM)

    def test_mcs_reproducibility_with_seed(self):
        """Test that MCS with same seed produces identical results."""
        sp = spec()
        mc_seed = MonteCarloParams(random_seed=123)
        pv1 = pv(_gbm(paths=1000), sp, PricingMethod.MONTE_CARLO, params=mc_seed)
        pv2 = pv(_gbm(paths=1000), sp, PricingMethod.MONTE_CARLO, params=mc_seed)
        assert np.isclose(pv1, pv2)

    def test_mcs_american_option(self):
        """Test MCS American option using Longstaff-Schwartz."""
        gbm = _gbm(paths=5000)
        am_spec = spec(OptionType.PUT, exercise=ExerciseType.AMERICAN)

        pv_deg2 = pv(
            gbm, am_spec, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42, deg=2)
        )
        pv_deg5 = pv(
            gbm, am_spec, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42, deg=5)
        )

        assert pv_deg2 > 0 and pv_deg5 > 0
        # Different regression degrees should remain close, but allow mild MC variance.
        assert np.isclose(pv_deg5, pv_deg2, rtol=_MC_RTOL_CROSS_METHOD)

        # binomial should also be close
        ud_bin = UnderlyingData(initial_value=SPOT, volatility=VOL, market_data=_market_data())
        pv_binom = pv(ud_bin, am_spec, PricingMethod.BINOMIAL, params=BINOM_PARAMS)
        assert np.isclose(pv_deg2, pv_binom, rtol=_MC_RTOL_CROSS_METHOD)

    def test_mcs_american_put_supports_immediate_exercise(self):
        """Deep ITM put with tiny vol and high rates should exercise immediately."""
        high_rate_curve = flat_curve(PRICING_DATE, MATURITY, 0.20)
        gbm = _gbm(paths=20_000, spot=40.0, vol=0.01, discount_curve=high_rate_curve)
        am_spec = spec(OptionType.PUT, exercise=ExerciseType.AMERICAN, strike=100.0)

        result = pv(
            gbm,
            am_spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=2),
        )

        assert np.isclose(result, 60.0, atol=0.01)

    def test_mcs_pathwise_return(self):
        """Test MCS present_value_pathwise returns discounted PVs per path."""
        gbm = _gbm(paths=1000)
        valuation = OptionValuation(gbm, spec(), PricingMethod.MONTE_CARLO, params=MC_PARAMS)

        pv_scalar = valuation.present_value()
        pv_pathwise = valuation.present_value_pathwise()

        assert isinstance(pv_scalar, (float, np.floating))
        assert isinstance(pv_pathwise, np.ndarray)
        assert pv_pathwise.shape[0] == 1000
        assert np.isclose(np.mean(pv_pathwise), pv_scalar)


def _plain_gbm(paths: int, *, antithetic: bool, vol: float = VOL) -> GBMProcess:
    """GBM with i.i.d. draws unless *antithetic*; moment matching off."""
    sim_config = SimulationConfig(
        paths=paths,
        num_steps=4,
        end_date=MATURITY,
        antithetic=antithetic,
        moment_matching=False,
    )
    return GBMProcess(_market_data(), GBMParams(initial_value=SPOT, volatility=vol), sim_config)


class TestMCStdError:
    """OptionValuation.std_error(): Monte Carlo only, calibrated, antithetic-aware."""

    @pytest.mark.parametrize(
        "method", [PricingMethod.ANALYTICAL, PricingMethod.BINOMIAL, PricingMethod.PDE_FD]
    )
    def test_none_for_deterministic_methods(self, method: PricingMethod):
        ud = UnderlyingData(initial_value=SPOT, volatility=VOL, market_data=_market_data())
        assert OptionValuation(ud, spec(), method).std_error() is None

    def test_positive_and_memoised_for_mc(self):
        ov = OptionValuation(
            _gbm(paths=20_000, num_steps=4), spec(), PricingMethod.MONTE_CARLO, params=MC_PARAMS
        )
        se = ov.std_error()
        assert se is not None and se > 0.0
        assert ov.std_error() == se

    def test_american_lsm_reports_std_error(self):
        ov = OptionValuation(
            _gbm(paths=20_000, num_steps=4),
            spec(OptionType.PUT, ExerciseType.AMERICAN),
            PricingMethod.MONTE_CARLO,
            params=MC_PARAMS,
        )
        se = ov.std_error()
        assert se is not None and se > 0.0

    @pytest.mark.parametrize("antithetic,strike", [(False, 100.0), (True, 60.0)])
    def test_matches_seed_to_seed_spread(self, antithetic: bool, strike: float):
        """Reported SE should agree with the empirical seed-to-seed PV dispersion.

        Across 20 independent seeds, the sample sd of the PVs should be close
        to the mean reported SE. If the SE were known exactly the ratio would
        be chi(19)/sqrt(19) (95% in [0.68, 1.31]); it is estimated, so the
        band is deliberately wider.

        The antithetic case uses a deep ITM call, nearly linear in S_T, so
        mirrored paths are strongly negatively correlated: the paired SE gives
        a ratio of 0.90 here, while treating mirrored paths as independent
        gives 0.18 and fails. An ATM call would not discriminate (unpaired
        ratio 0.61, inside the band).
        """
        process = _plain_gbm(20_000, antithetic=antithetic)
        pvs, ses = [], []
        for seed in range(1000, 1020):
            ov = OptionValuation(
                process,
                spec(strike=strike),
                PricingMethod.MONTE_CARLO,
                params=MonteCarloParams(random_seed=seed),
            )
            pvs.append(ov.present_value())
            ses.append(ov.std_error())
        ratio = np.std(pvs, ddof=1) / np.mean(ses)
        assert 0.6 < ratio < 1.4

    def test_european_within_a_few_std_errors_of_bsm(self):
        """Default MC settings (antithetic + moment matching) across seeds."""
        ud = UnderlyingData(initial_value=SPOT, volatility=VOL, market_data=_market_data())
        bsm = OptionValuation(ud, spec(), PricingMethod.ANALYTICAL).present_value()
        for seed in range(1000, 1005):
            ov = OptionValuation(
                _gbm(paths=20_000, num_steps=4),
                spec(),
                PricingMethod.MONTE_CARLO,
                params=MonteCarloParams(random_seed=seed),
            )
            se = ov.std_error()
            assert se is not None
            assert abs(ov.present_value() - bsm) < 4.0 * se

    def test_averages_antithetic_pairs(self):
        """Antithetic pairing cuts the SE of the payoff S_T by the expected factor.

        Under GBM, S_T = F e^{-x^2/2 + xZ} with x = sigma sqrt(T); a mirrored
        pair averages to F e^{-x^2/2} cosh(xZ). With n total paths (n/2
        pairs), plain sampling has SE^2 = Var(e^{xZ}) / n and paired
        sampling SE^2 = Var(cosh xZ) / (n/2), so the ratio is

            sqrt(2 Var(cosh xZ) / Var(e^{xZ})),
            Var(cosh xZ) = (1 + e^{2x^2}) / 2 - e^{x^2},
            Var(e^{xZ}) = e^{2x^2} - e^{x^2},

        about x for small x. Here sigma = 0.2 and T = 1, so x = 0.2 and the
        ratio is 0.198 (measured 0.197). Treating mirrored paths as
        independent would report roughly the plain SE (ratio ~1) instead.
        The 0.3 bound is a loose regression check.
        """
        payoff = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=MATURITY,
            payoff_fn=lambda s: s,
            currency=CURRENCY,
        )

        def se_with(antithetic: bool) -> float:
            process = _plain_gbm(20_000, antithetic=antithetic)
            se = OptionValuation(
                process, payoff, PricingMethod.MONTE_CARLO, params=MC_PARAMS
            ).std_error()
            assert se is not None
            return se

        assert se_with(True) < 0.3 * se_with(False)

    def test_nan_with_a_single_antithetic_pair(self):
        """One pair is one independent sample, so the ddof=1 variance is undefined."""
        ov = OptionValuation(
            _plain_gbm(2, antithetic=True), spec(), PricingMethod.MONTE_CARLO, params=MC_PARAMS
        )
        se = ov.std_error()
        assert se is not None and np.isnan(se)

    def test_rejects_control_variate(self):
        ov = OptionValuation(
            _gbm(paths=20_000, num_steps=4),
            spec(OptionType.PUT, ExerciseType.AMERICAN),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, control_variate_european=True),
        )
        with pytest.raises(UnsupportedFeatureError, match="std_error is not supported"):
            ov.std_error()

    def test_continuous_barrier_within_a_few_std_errors_of_closed_form(self):
        """Under GBM, Brownian-bridge survival weighting removes the discrete-monitoring
        bias, so MC and the closed form should agree within sampling error."""
        barrier = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=MATURITY,
            barrier=115.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            currency=CURRENCY,
        )
        ud = UnderlyingData(initial_value=SPOT, volatility=0.25, market_data=_market_data())
        closed_form = OptionValuation(ud, barrier, PricingMethod.ANALYTICAL).present_value()
        process = GBMProcess(
            _market_data(),
            GBMParams(initial_value=SPOT, volatility=0.25),
            SimulationConfig(paths=200_000, num_steps=26, end_date=MATURITY),
        )
        ov = OptionValuation(process, barrier, PricingMethod.MONTE_CARLO, params=MC_PARAMS)
        se = ov.std_error()
        assert se is not None and se > 0.0
        assert abs(ov.present_value() - closed_form) < 4.0 * se
