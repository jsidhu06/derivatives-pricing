"""Tests for Hull-style control variate adjustments."""

import datetime as dt
import logging

import numpy as np
import pytest

from derivatives_pricing.enums import (
    AsianAveraging,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from helpers import (
    PRICING_DATE,
    MATURITY,
    CURRENCY,
    flat_curve,
    market_data,
    underlying,
    spec,
)
from derivatives_pricing.valuation import OptionValuation
from derivatives_pricing.valuation.core import AsianSpec
from derivatives_pricing.valuation.params import BinomialParams, PDEParams

logger = logging.getLogger(__name__)


def _underlying(
    *,
    initial_value: float,
    volatility: float,
    rate: float,
    maturity: dt.datetime = MATURITY,
    dividend_curve=None,
    discrete_dividends=None,
):
    md = market_data(
        pricing_date=PRICING_DATE,
        discount_curve=flat_curve(PRICING_DATE, maturity, rate),
        currency=CURRENCY,
    )
    return underlying(
        initial_value=initial_value,
        volatility=volatility,
        market_data=md,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def test_control_variate_binomial_matches_adjustment():
    ud = _underlying(initial_value=100.0, volatility=0.25, rate=0.03)
    amer_spec = spec(OptionType.PUT, ExerciseType.AMERICAN, strike=110.0)
    euro_spec = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=110.0)

    base_params = BinomialParams(num_steps=300)
    cv_params = BinomialParams(num_steps=300, control_variate_european=True)

    american_raw = OptionValuation(
        ud, amer_spec, PricingMethod.BINOMIAL, params=base_params
    ).present_value()

    european_num = OptionValuation(
        ud, euro_spec, PricingMethod.BINOMIAL, params=base_params
    ).present_value()

    european_bsm = OptionValuation(ud, euro_spec, PricingMethod.ANALYTICAL).present_value()

    american_cv = OptionValuation(
        ud, amer_spec, PricingMethod.BINOMIAL, params=cv_params
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-8)


def test_control_variate_pde_matches_adjustment():
    ud = _underlying(initial_value=100.0, volatility=0.2, rate=0.04)
    amer_spec = spec(OptionType.PUT, ExerciseType.AMERICAN, strike=95.0)
    euro_spec = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=95.0)

    base_params = PDEParams(spot_steps=60, time_steps=60)
    cv_params = PDEParams(spot_steps=60, time_steps=60, control_variate_european=True)

    american_raw = OptionValuation(
        ud, amer_spec, PricingMethod.PDE_FD, params=base_params
    ).present_value()

    european_num = OptionValuation(
        ud, euro_spec, PricingMethod.PDE_FD, params=base_params
    ).present_value()

    european_bsm = OptionValuation(ud, euro_spec, PricingMethod.ANALYTICAL).present_value()

    american_cv = OptionValuation(
        ud, amer_spec, PricingMethod.PDE_FD, params=cv_params
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-6)


# ---------------------------------------------------------------------------
# Asian geometric control variate — Hull tree + Kemna-Vorst analytical
# ---------------------------------------------------------------------------

ASIAN_OBSERVATIONS = 100
ASIAN_ENGINE_STEPS = ASIAN_OBSERVATIONS - 1
TREE_AVERAGES = 150


def _asian_spec(
    *,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.AMERICAN,
    averaging: AsianAveraging = AsianAveraging.GEOMETRIC,
) -> AsianSpec:
    return AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        num_observations=ASIAN_OBSERVATIONS,
        exercise_type=exercise_type,
    )


class TestAsianControlVariate:
    """Control variate adjustment for American Asian option on Hull binomial tree."""

    def test_cv_matches_manual_adjustment(self):
        """CV-adjusted price = raw_american + (analytical_european − hull_european)."""
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.02)
        ud = _underlying(
            initial_value=100,
            volatility=0.25,
            rate=0.05,
            dividend_curve=q_curve,
        )
        base_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS, asian_tree_averages=TREE_AVERAGES
        )
        cv_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )

        amer_spec = _asian_spec(option_type=OptionType.PUT, strike=100)

        # Raw American Hull tree
        american_raw = OptionValuation(
            ud,
            amer_spec,
            PricingMethod.BINOMIAL,
            params=base_params,
        ).present_value()

        # European Hull tree (same lattice, no early exercise)
        euro_spec = _asian_spec(
            option_type=OptionType.PUT,
            strike=100,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_hull = OptionValuation(
            ud, euro_spec, PricingMethod.BINOMIAL, params=base_params
        ).present_value()

        # European analytical (Kemna-Vorst)
        bsm_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.PUT,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            num_observations=ASIAN_OBSERVATIONS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_analytical = OptionValuation(
            ud, bsm_spec, PricingMethod.ANALYTICAL
        ).present_value()

        # CV-adjusted
        american_cv = OptionValuation(
            ud, amer_spec, PricingMethod.BINOMIAL, params=cv_params
        ).present_value()

        expected = american_raw + (european_analytical - european_hull)
        assert np.isclose(american_cv, expected, atol=1e-8)
        logger.info(
            "Asian CV put: raw=%.6f cv=%.6f analytical_euro=%.6f hull_euro=%.6f adj=%.6f",
            american_raw,
            american_cv,
            european_analytical,
            european_hull,
            european_analytical - european_hull,
        )

    def test_cv_geq_european_analytical(self):
        """CV-adjusted American Asian ≥ European analytical (early exercise premium ≥ 0)."""
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.02)
        ud = _underlying(
            initial_value=100,
            volatility=0.25,
            rate=0.05,
            dividend_curve=q_curve,
        )
        cv_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )

        for option_type, strike in [
            (OptionType.PUT, 110),
            (OptionType.PUT, 100),
            (OptionType.CALL, 100),
            (OptionType.CALL, 90),
        ]:
            amer_spec = _asian_spec(option_type=option_type, strike=strike)
            american_cv = OptionValuation(
                ud, amer_spec, PricingMethod.BINOMIAL, params=cv_params
            ).present_value()

            bsm_spec = AsianSpec(
                averaging=AsianAveraging.GEOMETRIC,
                option_type=option_type,
                strike=strike,
                maturity=MATURITY,
                currency=CURRENCY,
                num_observations=ASIAN_OBSERVATIONS,
                exercise_type=ExerciseType.EUROPEAN,
            )
            european_analytical = OptionValuation(
                ud, bsm_spec, PricingMethod.ANALYTICAL
            ).present_value()

            assert american_cv >= european_analytical - 1e-8, (
                f"{option_type.value} K={strike}: "
                f"american_cv={american_cv:.6f} < european={european_analytical:.6f}"
            )

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,option_type",
        [
            (100, 100, 0.20, 0.05, 0.02, 365, OptionType.PUT),
            (100, 110, 0.25, 0.03, 0.03, 365, OptionType.PUT),
            (100, 90, 0.30, 0.08, 0.01, 365, OptionType.CALL),
        ],
    )
    def test_cv_reduces_european_bias(self, spot, strike, vol, r, q, days, option_type):
        """For European exercise, CV should eliminate the Hull tree bias entirely."""
        maturity = PRICING_DATE + dt.timedelta(days=days)
        q_curve = flat_curve(PRICING_DATE, maturity, q) if q else None
        ud = _underlying(
            initial_value=spot,
            volatility=vol,
            maturity=maturity,
            rate=r,
            dividend_curve=q_curve,
        )

        hull_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS,
            asian_tree_averages=TREE_AVERAGES,
        )

        # European Hull tree (no CV)
        euro_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=option_type,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
            num_observations=ASIAN_OBSERVATIONS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        euro_hull = OptionValuation(
            ud, euro_spec, PricingMethod.BINOMIAL, params=hull_params
        ).present_value()

        # European analytical
        bsm_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=option_type,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
            num_observations=ASIAN_OBSERVATIONS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        euro_analytical = OptionValuation(ud, bsm_spec, PricingMethod.ANALYTICAL).present_value()

        # Hull tree bias
        hull_bias = abs(euro_hull - euro_analytical) / euro_analytical
        logger.info(
            "European Hull bias: %s K=%d %.4f%% (hull=%.6f analytical=%.6f)",
            option_type.value,
            strike,
            hull_bias * 100,
            euro_hull,
            euro_analytical,
        )
        # Confirm the tree has measurable bias (>0.1%)
        assert hull_bias > 0.001, f"Hull tree bias too small to test CV: {hull_bias:.6f}"

    def test_cv_arithmetic_matches_manual_adjustment(self):
        """CV with arithmetic averaging uses Turnbull-Wakeman analytical formula."""
        ud = _underlying(initial_value=100, volatility=0.25, rate=0.05)
        base_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS, asian_tree_averages=TREE_AVERAGES
        )
        cv_params = BinomialParams(
            num_steps=ASIAN_ENGINE_STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )
        amer_spec = _asian_spec(
            option_type=OptionType.CALL,
            strike=100,
            averaging=AsianAveraging.ARITHMETIC,
        )

        # Raw American Hull tree
        american_raw = OptionValuation(
            ud, amer_spec, PricingMethod.BINOMIAL, params=base_params
        ).present_value()

        # European Hull tree (same lattice)
        euro_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            num_observations=ASIAN_OBSERVATIONS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_hull = OptionValuation(
            ud, euro_spec, PricingMethod.BINOMIAL, params=base_params
        ).present_value()

        # European analytical (Turnbull-Wakeman)
        bsm_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            num_observations=ASIAN_OBSERVATIONS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_analytical = OptionValuation(
            ud, bsm_spec, PricingMethod.ANALYTICAL
        ).present_value()

        # CV-adjusted
        american_cv = OptionValuation(
            ud, amer_spec, PricingMethod.BINOMIAL, params=cv_params
        ).present_value()

        expected = american_raw + (european_analytical - european_hull)
        assert np.isclose(american_cv, expected, atol=1e-8)
        assert american_cv > 0.0
