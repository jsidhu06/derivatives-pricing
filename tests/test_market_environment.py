"""Tests for MarketData and CorrelationContext validation."""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import DayCountConvention
from derivatives_pricing.exceptions import ValidationError
from derivatives_pricing.market_environment import CorrelationContext, MarketData
from derivatives_pricing.rates import DiscountCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
CURVE = DiscountCurve.flat(0.05, end_time=1.0)


# ---------------------------------------------------------------------------
# MarketData
# ---------------------------------------------------------------------------


class TestMarketDataValidation:
    def test_valid_construction(self):
        md = MarketData(PRICING_DATE, CURVE, currency="USD")
        assert md.pricing_date == PRICING_DATE
        assert md.currency == "USD"
        assert md.day_count_convention is DayCountConvention.ACT_365F

    def test_custom_day_count(self):
        md = MarketData(
            PRICING_DATE, CURVE, currency="USD", day_count_convention=DayCountConvention.ACT_360
        )
        assert md.day_count_convention is DayCountConvention.ACT_360

    def test_rejects_bad_pricing_date(self):
        with pytest.raises(ValidationError, match="pricing_date must be a datetime"):
            MarketData("2025-01-01", CURVE, currency="USD")

    def test_rejects_bad_discount_curve(self):
        with pytest.raises(ValidationError, match="discount_curve must be a DiscountCurve"):
            MarketData(PRICING_DATE, 0.05, currency="USD")

    def test_rejects_empty_currency(self):
        with pytest.raises(ValidationError, match="currency must be a non-empty string"):
            MarketData(PRICING_DATE, CURVE, currency="")

    def test_rejects_non_string_currency(self):
        with pytest.raises(ValidationError, match="currency must be a non-empty string"):
            MarketData(PRICING_DATE, CURVE, currency=123)

    def test_rejects_bad_day_count_convention(self):
        with pytest.raises(ValidationError, match="day_count_convention must be"):
            MarketData(PRICING_DATE, CURVE, currency="USD", day_count_convention="ACT_365F")


# ---------------------------------------------------------------------------
# CorrelationContext
# ---------------------------------------------------------------------------


def _grid(n_intervals: int) -> list[dt.datetime]:
    """Monthly-ish grid from PRICING_DATE with *n_intervals* steps."""
    return [PRICING_DATE + dt.timedelta(days=30 * k) for k in range(n_intervals + 1)]


def _valid_corr_context(**overrides):
    """Build a valid 2-asset CorrelationContext, with optional overrides."""
    defaults = dict(
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
        random_numbers=np.random.default_rng(0).standard_normal((2, 10, 100)),
        asset_names=["A", "B"],
        time_grid=_grid(10),
    )
    defaults.update(overrides)
    return CorrelationContext(**defaults)


class TestCorrelationContextValid:
    def test_valid_construction(self):
        ctx = _valid_corr_context()
        assert ctx.cholesky_matrix.shape == (2, 2)
        assert ctx.asset_index("A") == 0
        assert ctx.asset_index("B") == 1

    def test_cholesky_reconstructs_correlation(self):
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        ctx = _valid_corr_context(correlation_matrix=C)
        np.testing.assert_allclose(ctx.cholesky_matrix @ ctx.cholesky_matrix.T, C)

    def test_four_asset_context_with_zero_entries(self):
        C = np.array(
            [
                [1.0, 0.0, 0.3, -0.2],
                [0.0, 1.0, -0.1, 0.25],
                [0.3, -0.1, 1.0, 0.0],
                [-0.2, 0.25, 0.0, 1.0],
            ]
        )
        rn = np.random.default_rng(0).standard_normal((4, 8, 64))
        ctx = CorrelationContext(
            correlation_matrix=C,
            random_numbers=rn,
            asset_names=["A", "B", "C", "D"],
            time_grid=_grid(8),
        )
        np.testing.assert_allclose(
            ctx.cholesky_matrix @ ctx.cholesky_matrix.T, C, rtol=1e-10, atol=1e-10
        )

    def test_near_singular_matrix_still_constructs(self):
        """Matrix that is very close to singular but SPD should still be accepted."""
        rho = 0.999
        C = np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])
        rn = np.random.default_rng(1).standard_normal((3, 6, 32))
        ctx = CorrelationContext(
            correlation_matrix=C, random_numbers=rn, asset_names=["A", "B", "C"], time_grid=_grid(6)
        )
        np.testing.assert_allclose(
            ctx.cholesky_matrix @ ctx.cholesky_matrix.T, C, rtol=1e-8, atol=1e-8
        )


class TestCorrelationContextValidation:
    def test_rejects_non_square_matrix(self):
        with pytest.raises(ValidationError, match="must be square"):
            _valid_corr_context(correlation_matrix=np.ones((2, 3)))

    def test_rejects_1d_matrix(self):
        with pytest.raises(ValidationError, match="must be square"):
            _valid_corr_context(correlation_matrix=np.ones(4))

    def test_rejects_mismatched_asset_names_length(self):
        with pytest.raises(ValidationError, match="asset_names length"):
            _valid_corr_context(asset_names=["A", "B", "C"])

    def test_rejects_duplicate_asset_names(self):
        with pytest.raises(ValidationError, match="unique entries"):
            _valid_corr_context(asset_names=["A", "A"])

    def test_rejects_asymmetric_matrix(self):
        C = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValidationError, match="symmetric"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_non_unit_diagonal(self):
        C = np.array([[2.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValidationError, match="diagonal must be 1"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_non_positive_definite(self):
        # Correlation > 1 in off-diag makes it not positive-definite
        C = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValidationError, match="not positive-definite"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_2d_random_numbers(self):
        rn = np.ones((2, 10))
        with pytest.raises(ValidationError, match="must be 3-D"):
            _valid_corr_context(random_numbers=rn)

    def test_rejects_wrong_random_numbers_axis0(self):
        rn = np.ones((3, 10, 100))  # 3 != 2 assets
        with pytest.raises(ValidationError, match="axis-0"):
            _valid_corr_context(random_numbers=rn)

    def test_unknown_asset_name_raises(self):
        ctx = _valid_corr_context()
        with pytest.raises(ValidationError, match="not found in asset_names"):
            ctx.asset_index("C")

    def test_rejects_singular_correlation_matrix(self):
        C = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rn = np.random.default_rng(0).standard_normal((3, 6, 64))
        with pytest.raises(ValidationError, match="not positive-definite"):
            CorrelationContext(
                correlation_matrix=C,
                random_numbers=rn,
                asset_names=["A", "B", "C"],
                time_grid=_grid(6),
            )


class TestCorrelationContextTimeGrid:
    def test_grid_is_stored_as_a_tuple(self):
        ctx = _valid_corr_context()
        assert ctx.time_grid == tuple(_grid(10))
        assert ctx.paths == 100

    def test_rejects_random_numbers_not_matching_the_grid(self):
        with pytest.raises(ValidationError, match=r"axis-1 \(10\) must equal .* intervals \(9\)"):
            _valid_corr_context(time_grid=_grid(9))

    def test_rejects_unsorted_grid(self):
        grid = _grid(10)
        grid[3], grid[4] = grid[4], grid[3]
        with pytest.raises(ValidationError, match="strictly ascending"):
            _valid_corr_context(time_grid=grid)

    def test_rejects_single_date_grid(self):
        with pytest.raises(ValidationError, match="at least two dates"):
            _valid_corr_context(time_grid=[PRICING_DATE])

    def test_from_seed_draws_matching_shapes(self):
        ctx = CorrelationContext.from_seed(
            correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            asset_names=["A", "B"],
            time_grid=_grid(12),
            paths=64,
            random_seed=7,
        )
        assert ctx.random_numbers.shape == (2, 12, 64)
        again = CorrelationContext.from_seed(
            correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            asset_names=["A", "B"],
            time_grid=_grid(12),
            paths=64,
            random_seed=7,
        )
        np.testing.assert_array_equal(ctx.random_numbers, again.random_numbers)
