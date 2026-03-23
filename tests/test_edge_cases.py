"""Edge-case tests: zero vol, near-zero expiry, deep ITM/OTM, extreme rates.

These tests verify that the library behaves correctly (or fails gracefully)
at the boundaries of its input space.
"""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import (
    ExerciseType,
    OptionType,
    PDESpaceGrid,
    PricingMethod,
)
from derivatives_pricing.exceptions import (
    ArbitrageViolationError,
    NumericalError,
    ValidationError,
)
from helpers import (
    flat_curve,
    market_data,
    underlying,
    make_vanilla_spec,
    pv as _pv,
)
from derivatives_pricing.valuation import (
    BinomialParams,
    VanillaSpec,
    PDEParams,
    UnderlyingData,
)

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2025, 7, 3)  # ~0.5y
RATE = 0.05
_DEFAULT_RATE_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)


def _underlying(
    spot: float = 100.0,
    vol: float = 0.20,
    discount_curve=None,
    dividend_curve=None,
) -> UnderlyingData:
    rate_curve = discount_curve if discount_curve is not None else _DEFAULT_RATE_CURVE
    return underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data(
            pricing_date=PRICING_DATE,
            discount_curve=rate_curve,
            currency="USD",
        ),
        dividend_curve=dividend_curve,
    )


def _spec(
    strike: float = 100.0,
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    maturity: dt.datetime = MATURITY,
) -> VanillaSpec:
    return make_vanilla_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=exercise,
        currency="USD",
    )


# ═══════════════════════════════════════════════════════════════════════
#  Zero volatility
# ═══════════════════════════════════════════════════════════════════════


class TestZeroVolatility:
    """σ = 0 ⇒ deterministic world; option value = discounted intrinsic."""

    # --- BSM: zero-vol guard returns ±inf for d1/d2, yielding intrinsic ---

    @pytest.mark.parametrize(
        "option_type,strike,expected_positive",
        [
            (OptionType.CALL, 90.0, True),  # ITM call
            (OptionType.PUT, 110.0, True),  # ITM put
            (OptionType.CALL, 110.0, False),  # OTM call
            (OptionType.PUT, 90.0, False),  # OTM put
        ],
    )
    def test_bsm_zero_vol_pricing(self, option_type, strike, expected_positive):
        """BSM with vol=0 should still price correctly (discounted intrinsic)."""
        spot = 100.0
        ud = _underlying(spot=spot, vol=0.0)
        spec = _spec(strike=strike, option_type=option_type)
        pv = _pv(ud, spec, PricingMethod.BSM)

        if expected_positive:
            # Zero-vol BSM: Call = S*df_q - K*df_r, Put = K*df_r - S*df_q
            # With no dividends df_q = 1.
            df_r = float(ud.discount_curve.df(0.5))
            if option_type is OptionType.CALL:
                expected = spot - strike * df_r
            else:
                expected = strike * df_r - spot
            assert pv > 0
            assert np.isclose(pv, expected, rtol=0.01)
        else:
            assert np.isclose(pv, 0.0, atol=1e-10)

    def test_bsm_zero_vol_atm(self):
        """BSM with vol=0, ATM ⇒ value is 0 for both call and put (no time value)."""
        ud = _underlying(vol=0.0)
        for opt in (OptionType.CALL, OptionType.PUT):
            spec = _spec(strike=100.0, option_type=opt)
            pv = _pv(ud, spec, PricingMethod.BSM)
            # ATM with zero vol: forward = S*exp((r-q)*T), if forward == strike exactly
            # then it could be slightly non-zero due to discounting. Check it's small.
            assert abs(pv) < 5.0  # no large time value

    # --- Binomial: vol=0 makes u=d=1, violating d < growth < u ---

    def test_binomial_zero_vol_raises(self):
        """CRR tree with vol=0 should raise ArbitrageViolationError (u=d=1)."""
        ud = _underlying(vol=0.0)
        spec = _spec(strike=90.0, option_type=OptionType.CALL)
        with pytest.raises(NumericalError):
            _pv(ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=100))

    # --- PDE: explicit vol > 0 guard ---

    def test_pde_zero_vol_raises(self):
        """PDE solver explicitly validates vol > 0."""
        ud = _underlying(vol=0.0)
        spec = _spec(strike=90.0, option_type=OptionType.CALL)
        with pytest.raises(ValidationError, match="volatility must be positive"):
            _pv(ud, spec, PricingMethod.PDE_FD)


# ═══════════════════════════════════════════════════════════════════════
#  Very low volatility (non-zero, but tiny)
# ═══════════════════════════════════════════════════════════════════════


class TestLowVolatility:
    """Tiny σ → option value converges to discounted intrinsic."""

    VOL = 1e-4  # 0.01%

    @pytest.mark.parametrize(
        "option_type,strike",
        [
            (OptionType.CALL, 80.0),
            (OptionType.PUT, 120.0),
        ],
    )
    def test_bsm_low_vol_itm_near_intrinsic(self, option_type, strike):
        """Deep ITM with tiny vol should be very close to discounted intrinsic."""
        spot = 100.0
        ud = _underlying(spot=spot, vol=self.VOL)
        spec = _spec(strike=strike, option_type=option_type)
        pv = _pv(ud, spec, PricingMethod.BSM)

        df_r = float(ud.discount_curve.df(0.5))
        # With near-zero vol: Call ≈ S - K*df_r, Put ≈ K*df_r - S
        if option_type is OptionType.CALL:
            expected = spot - strike * df_r
        else:
            expected = strike * df_r - spot
        assert np.isclose(pv, expected, rtol=0.01)

    @pytest.mark.parametrize(
        "option_type,strike",
        [
            (OptionType.CALL, 120.0),
            (OptionType.PUT, 80.0),
        ],
    )
    def test_bsm_low_vol_otm_near_zero(self, option_type, strike):
        """OTM with tiny vol has negligible probability of exercise."""
        ud = _underlying(vol=self.VOL)
        spec = _spec(strike=strike, option_type=option_type)
        pv = _pv(ud, spec, PricingMethod.BSM)
        assert pv < 0.01


# ═══════════════════════════════════════════════════════════════════════
#  Near-zero expiry (1 day)
# ═══════════════════════════════════════════════════════════════════════


class TestNearZeroExpiry:
    """T → 0 ⇒ option value → intrinsic (nearly undiscounted)."""

    SHORT_MATURITY = PRICING_DATE + dt.timedelta(days=1)

    @pytest.mark.parametrize(
        "option_type,strike,expected_intrinsic",
        [
            (OptionType.CALL, 90.0, 10.0),
            (OptionType.PUT, 110.0, 10.0),
            (OptionType.CALL, 110.0, 0.0),
            (OptionType.PUT, 90.0, 0.0),
        ],
    )
    def test_bsm_near_expiry(self, option_type, strike, expected_intrinsic):
        """BSM with 1-day expiry should be very close to intrinsic."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, self.SHORT_MATURITY, RATE))
        spec = _spec(strike=strike, option_type=option_type, maturity=self.SHORT_MATURITY)
        pv = _pv(ud, spec, PricingMethod.BSM)
        assert np.isclose(pv, expected_intrinsic, atol=0.50)

    @pytest.mark.parametrize(
        "option_type,strike,expected_intrinsic",
        [
            (OptionType.CALL, 90.0, 10.0),
            (OptionType.PUT, 110.0, 10.0),
            (OptionType.CALL, 110.0, 0.0),
            (OptionType.PUT, 90.0, 0.0),
        ],
    )
    def test_binomial_near_expiry(self, option_type, strike, expected_intrinsic):
        """Binomial with 1-day expiry should be very close to intrinsic."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, self.SHORT_MATURITY, RATE))
        spec = _spec(strike=strike, option_type=option_type, maturity=self.SHORT_MATURITY)
        pv = _pv(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=50),
        )
        assert np.isclose(pv, expected_intrinsic, atol=0.50)

    @pytest.mark.parametrize(
        "option_type,strike,expected_intrinsic",
        [
            (OptionType.CALL, 90.0, 10.0),
            (OptionType.PUT, 110.0, 10.0),
            (OptionType.CALL, 110.0, 0.0),
            (OptionType.PUT, 90.0, 0.0),
        ],
    )
    def test_pde_near_expiry(self, option_type, strike, expected_intrinsic):
        """PDE with 1-day expiry should be very close to intrinsic."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, self.SHORT_MATURITY, RATE))
        spec = _spec(strike=strike, option_type=option_type, maturity=self.SHORT_MATURITY)
        pv = _pv(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=200, time_steps=50),
        )
        assert np.isclose(pv, expected_intrinsic, atol=0.50)


# ═══════════════════════════════════════════════════════════════════════
#  Deep ITM / OTM
# ═══════════════════════════════════════════════════════════════════════


class TestDeepMoneyness:
    """Very deep ITM ⇒ ~discounted forward intrinsic; deep OTM ⇒ ~0."""

    @staticmethod
    def _params(method: PricingMethod) -> dict:
        if method is PricingMethod.BINOMIAL:
            return {"params": BinomialParams(num_steps=200)}
        return {}

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_deep_itm_call(self, method):
        """Deep ITM call (S=200, K=50) should ≈ S*df_q - K*df_r."""
        spot, strike = 200.0, 50.0
        ud = _underlying(spot=spot, vol=0.20)
        spec = _spec(strike=strike, option_type=OptionType.CALL)
        pv = _pv(ud, spec, method, **self._params(method))

        df_r = float(ud.discount_curve.df(0.5))
        expected = spot - strike * df_r  # no dividends ⇒ df_q = 1
        assert np.isclose(pv, expected, rtol=0.01)

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_deep_otm_call(self, method):
        """Deep OTM call (S=50, K=200) should be nearly zero."""
        ud = _underlying(spot=50.0, vol=0.20)
        spec = _spec(strike=200.0, option_type=OptionType.CALL)
        pv = _pv(ud, spec, method, **self._params(method))
        assert pv < 0.01

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_deep_itm_put(self, method):
        """Deep ITM put (S=50, K=200) should ≈ K*df_r - S*df_q."""
        spot, strike = 50.0, 200.0
        ud = _underlying(spot=spot, vol=0.20)
        spec = _spec(strike=strike, option_type=OptionType.PUT)
        pv = _pv(ud, spec, method, **self._params(method))

        df_r = float(ud.discount_curve.df(0.5))
        expected = strike * df_r - spot
        assert np.isclose(pv, expected, rtol=0.01)

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_deep_otm_put(self, method):
        """Deep OTM put (S=200, K=50) should be nearly zero."""
        ud = _underlying(spot=200.0, vol=0.20)
        spec = _spec(strike=50.0, option_type=OptionType.PUT)
        pv = _pv(ud, spec, method, **self._params(method))
        assert pv < 0.01

    def test_pde_deep_itm_call(self):
        """PDE deep ITM call (log-spot grid handles extreme moneyness better)."""
        spot, strike = 200.0, 50.0
        ud = _underlying(spot=spot, vol=0.20)
        spec = _spec(strike=strike, option_type=OptionType.CALL)
        pv = _pv(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(
                spot_steps=300,
                time_steps=200,
                space_grid=PDESpaceGrid.LOG_SPOT,
            ),
        )
        df_r = float(ud.discount_curve.df(0.5))
        expected = spot - strike * df_r
        assert np.isclose(pv, expected, rtol=0.02)

    def test_pde_deep_otm_call(self):
        """PDE deep OTM call should be nearly zero."""
        ud = _underlying(spot=50.0, vol=0.20)
        spec = _spec(strike=200.0, option_type=OptionType.CALL)
        pv = _pv(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(
                spot_steps=300,
                time_steps=200,
                space_grid=PDESpaceGrid.LOG_SPOT,
            ),
        )
        assert pv < 0.50  # log-spot grid may have some residual


# ═══════════════════════════════════════════════════════════════════════
#  Extreme interest rates
# ═══════════════════════════════════════════════════════════════════════


class TestExtremeRates:
    """Very high or negative risk-free rates."""

    @staticmethod
    def _params(method: PricingMethod) -> dict:
        if method is PricingMethod.BINOMIAL:
            return {"params": BinomialParams(num_steps=200)}
        return {}

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_high_rate_call_value_increases(self, method):
        """Higher r increases call value (forward price is higher)."""
        spec = _spec(strike=100.0, option_type=OptionType.CALL)
        ud_low = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.01))
        ud_high = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.30))

        pv_low = _pv(ud_low, spec, method, **self._params(method))
        pv_high = _pv(ud_high, spec, method, **self._params(method))
        assert pv_high > pv_low

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_high_rate_put_value_decreases(self, method):
        """Higher r decreases put value (forward price is higher)."""
        spec = _spec(strike=100.0, option_type=OptionType.PUT)
        ud_low = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.01))
        ud_high = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.30))

        pv_low = _pv(ud_low, spec, method, **self._params(method))
        pv_high = _pv(ud_high, spec, method, **self._params(method))
        assert pv_high < pv_low

    def test_negative_rate_bsm(self):
        """Negative rates should produce valid (positive) option prices."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, -0.02))
        for opt in (OptionType.CALL, OptionType.PUT):
            spec = _spec(strike=100.0, option_type=opt)
            pv = _pv(ud, spec, PricingMethod.BSM)
            assert pv > 0
            assert np.isfinite(pv)

    def test_negative_rate_binomial(self):
        """Negative rates should produce valid prices in binomial model."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, -0.02))
        for opt in (OptionType.CALL, OptionType.PUT):
            spec = _spec(strike=100.0, option_type=opt)
            pv = _pv(
                ud,
                spec,
                PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=200),
            )
            assert pv > 0
            assert np.isfinite(pv)

    def test_negative_rate_pde(self):
        """Negative rates should produce valid prices in PDE solver."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, -0.02))
        for opt in (OptionType.CALL, OptionType.PUT):
            spec = _spec(strike=100.0, option_type=opt)
            pv = _pv(
                ud,
                spec,
                PricingMethod.PDE_FD,
                params=PDEParams(spot_steps=200, time_steps=200),
            )
            assert pv > 0
            assert np.isfinite(pv)

    def test_very_high_rate_bsm_convergence(self):
        """r = 50% should still produce finite BSM prices."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.50))
        spec = _spec(strike=100.0, option_type=OptionType.CALL)
        pv = _pv(ud, spec, PricingMethod.BSM)
        assert np.isfinite(pv)
        assert pv > 0

    def test_bsm_and_binomial_agree_at_extreme_rate(self):
        """BSM and binomial should agree even at r = 25%."""
        ud = _underlying(discount_curve=flat_curve(PRICING_DATE, MATURITY, 0.25))
        spec = _spec(strike=100.0, option_type=OptionType.CALL)
        bsm_pv = _pv(ud, spec, PricingMethod.BSM)
        bin_pv = _pv(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )
        assert np.isclose(bsm_pv, bin_pv, rtol=0.01)


# ═══════════════════════════════════════════════════════════════════════
#  High volatility
# ═══════════════════════════════════════════════════════════════════════


class TestHighVolatility:
    """σ = 200% — extreme but valid."""

    @staticmethod
    def _params(method: PricingMethod) -> dict:
        if method is PricingMethod.BINOMIAL:
            return {"params": BinomialParams(num_steps=200)}
        return {}

    @pytest.mark.parametrize("method", [PricingMethod.BSM, PricingMethod.BINOMIAL])
    def test_high_vol_produces_finite_prices(self, method):
        """Very high vol should still yield finite, positive option prices."""
        ud = _underlying(vol=2.0)
        for opt in (OptionType.CALL, OptionType.PUT):
            spec = _spec(strike=100.0, option_type=opt)
            pv = _pv(ud, spec, method, **self._params(method))
            assert np.isfinite(pv)
            assert pv > 0

    def test_high_vol_call_exceeds_low_vol_call(self):
        """Higher vol increases option value (more upside potential)."""
        spec = _spec(strike=100.0, option_type=OptionType.CALL)
        pv_low = _pv(_underlying(vol=0.10), spec, PricingMethod.BSM)
        pv_high = _pv(_underlying(vol=2.00), spec, PricingMethod.BSM)
        assert pv_high > pv_low

    def test_pde_high_vol_log_spot(self):
        """PDE with high vol should work on log-spot grid."""
        ud = _underlying(vol=1.50)
        spec = _spec(strike=100.0, option_type=OptionType.CALL)
        pv = _pv(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(
                spot_steps=300,
                time_steps=300,
                space_grid=PDESpaceGrid.LOG_SPOT,
            ),
        )
        bsm_pv = _pv(ud, spec, PricingMethod.BSM)
        assert np.isclose(pv, bsm_pv, rtol=0.02)


# ═══════════════════════════════════════════════════════════════════════
#  American edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestAmericanEdgeCases:
    """American-specific edge cases at boundary conditions."""

    def test_american_call_no_dividend_equals_european(self):
        """American call with no dividends should equal European call."""
        ud = _underlying()
        spec_eu = _spec(option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)
        spec_am = _spec(option_type=OptionType.CALL, exercise=ExerciseType.AMERICAN)

        eu_pv = _pv(ud, spec_eu, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        am_pv = _pv(ud, spec_am, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        assert np.isclose(eu_pv, am_pv, rtol=1e-6)

    def test_american_put_at_least_european(self):
        """American put should always be >= European put."""
        ud = _underlying()
        spec_eu = _spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.EUROPEAN)
        spec_am = _spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)

        eu_pv = _pv(ud, spec_eu, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        am_pv = _pv(ud, spec_am, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        assert am_pv >= eu_pv - 1e-10

    def test_deep_itm_american_put_near_intrinsic(self):
        """Deep ITM American put should be at least intrinsic (early exercise)."""
        spot, strike = 50.0, 200.0
        ud = _underlying(spot=spot, vol=0.20)
        spec = _spec(strike=strike, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)
        pv = _pv(ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        intrinsic = strike - spot
        assert pv >= intrinsic - 0.01

    @pytest.mark.parametrize("spot,strike", [(40.0, 200.0), (25.0, 180.0), (10.0, 150.0)])
    def test_deep_itm_american_put_exceeds_intrinsic(self, spot, strike):
        """Deep ITM American put should be at least intrinsic value."""
        ud = _underlying(spot=spot, vol=0.2)
        spec = _spec(strike=strike, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)
        pv = _pv(ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        intrinsic = strike - spot
        assert pv >= intrinsic - 1e-8

    @pytest.mark.parametrize("spot,strike", [(120.0, 80.0), (150.0, 90.0), (200.0, 100.0)])
    def test_deep_itm_american_call_with_dividend_exceeds_european(self, spot, strike):
        """With a positive dividend yield, American call should dominate European."""
        ud = _underlying(
            spot=spot,
            vol=0.2,
            dividend_curve=flat_curve(PRICING_DATE, MATURITY, 0.05),
        )
        spec_eu = _spec(strike=strike, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)
        spec_am = _spec(strike=strike, option_type=OptionType.CALL, exercise=ExerciseType.AMERICAN)
        eu_pv = _pv(ud, spec_eu, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        am_pv = _pv(ud, spec_am, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))
        assert am_pv >= eu_pv - 1e-8

    @pytest.mark.parametrize("vol", [1.0e-4, 5.0e-4, 1.0e-3])
    def test_american_put_near_zero_vol_rejects_arbitrage_violation(self, vol):
        """Near-zero vol can violate CRR no-arbitrage bounds for fixed dt; raise cleanly."""
        spot, strike = 50.0, 200.0
        ud = _underlying(spot=spot, vol=vol)
        spec = _spec(strike=strike, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)
        with pytest.raises(ArbitrageViolationError, match="No-arbitrage condition"):
            _pv(ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500))

    @pytest.mark.parametrize("days", [1, 2, 3])
    def test_american_put_near_expiry_converges_to_intrinsic(self, days):
        """Near maturity, American put should be very close to intrinsic."""
        maturity = PRICING_DATE + dt.timedelta(days=days)
        spot, strike = 80.0, 100.0
        ud = _underlying(
            spot=spot,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, RATE),
        )
        spec = _spec(
            strike=strike,
            option_type=OptionType.PUT,
            exercise=ExerciseType.AMERICAN,
            maturity=maturity,
        )
        pv = _pv(ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=100))
        intrinsic = strike - spot
        assert np.isclose(pv, intrinsic, atol=0.5)

    def test_american_deep_itm_put_pde(self):
        """PDE American deep ITM put should also be >= intrinsic."""
        spot, strike = 50.0, 200.0
        ud = _underlying(spot=spot, vol=0.20)
        spec = _spec(strike=strike, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)
        pv = _pv(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=300, time_steps=300),
        )
        intrinsic = strike - spot
        assert pv >= intrinsic - 0.01
