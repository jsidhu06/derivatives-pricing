"""Tests for Monte Carlo Simulation valuation."""

import numpy as np
import pytest

from derivatives_pricing.enums import ExerciseType, OptionType, PricingMethod
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
from helpers import flat_curve, market_data, pv, spec
from derivatives_pricing.valuation import (
    MonteCarloParams,
    OptionValuation,
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
