"""Tests for MonteCarloParams, BinomialParams, and PDEParams validation."""

import warnings

import pytest

from derivatives_pricing.enums import PDEMethod, PDESpaceGrid
from derivatives_pricing.exceptions import ValidationError
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams, PDEParams


# ---------------------------------------------------------------------------
# MonteCarloParams
# ---------------------------------------------------------------------------


class TestMonteCarloParams:
    def test_defaults_valid(self):
        p = MonteCarloParams()
        assert p.deg == 3
        assert p.ridge_lambda == 1e-8
        assert p.min_itm == 25
        assert p.barrier_aware_basis is True

    def test_rejects_deg_below_1(self):
        with pytest.raises(ValidationError, match="deg must be >= 1"):
            MonteCarloParams(deg=0)

    def test_rejects_negative_ridge_lambda(self):
        with pytest.raises(ValidationError, match="ridge_lambda must be >= 0"):
            MonteCarloParams(ridge_lambda=-1.0)

    def test_rejects_min_itm_below_1(self):
        with pytest.raises(ValidationError, match="min_itm must be >= 1"):
            MonteCarloParams(min_itm=0)

    def test_rejects_non_positive_std_error_warn_ratio(self):
        with pytest.raises(ValidationError, match="std_error_warn_ratio must be > 0"):
            MonteCarloParams(std_error_warn_ratio=-0.1)

    def test_zero_std_error_warn_ratio_rejected(self):
        with pytest.raises(ValidationError, match="std_error_warn_ratio must be > 0"):
            MonteCarloParams(std_error_warn_ratio=0.0)

    def test_none_std_error_warn_ratio_accepted(self):
        p = MonteCarloParams(std_error_warn_ratio=None)
        assert p.std_error_warn_ratio is None

    def test_rejects_non_bool_barrier_aware_basis(self):
        with pytest.raises(ValidationError, match="barrier_aware_basis must be a bool"):
            MonteCarloParams(barrier_aware_basis=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field,value", [("deg", 3.0), ("min_itm", 25.0)])
    def test_rejects_float_for_int_fields(self, field, value):
        with pytest.raises(ValidationError, match=f"{field} must be an int"):
            MonteCarloParams(**{field: value})

    @pytest.mark.parametrize("field,value", [("deg", True), ("min_itm", False)])
    def test_rejects_bool_for_int_fields(self, field, value):
        with pytest.raises(ValidationError, match=f"{field} must be an int"):
            MonteCarloParams(**{field: value})


# ---------------------------------------------------------------------------
# BinomialParams
# ---------------------------------------------------------------------------


class TestBinomialParams:
    def test_defaults_valid(self):
        p = BinomialParams()
        assert p.num_steps == 500
        assert p.mc_paths is None

    def test_rejects_num_steps_below_1(self):
        with pytest.raises(ValidationError, match="num_steps must be >= 1"):
            BinomialParams(num_steps=0)

    def test_rejects_both_mc_paths_and_asian_tree_averages(self):
        with pytest.raises(ValidationError, match="Only one of mc_paths and asian_tree_averages"):
            BinomialParams(mc_paths=1000, asian_tree_averages=50)

    def test_rejects_mc_paths_below_1(self):
        with pytest.raises(ValidationError, match="mc_paths must be >= 1"):
            BinomialParams(mc_paths=0)

    def test_rejects_asian_tree_averages_below_1(self):
        with pytest.raises(ValidationError, match="asian_tree_averages must be >= 1"):
            BinomialParams(asian_tree_averages=0)

    def test_warns_small_asian_tree_averages_ratio(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BinomialParams(num_steps=100, asian_tree_averages=10)
        assert any("interpolation bias" in str(warning.message) for warning in w)

    def test_warns_large_asian_tree_averages_ratio(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BinomialParams(num_steps=10, asian_tree_averages=30)
        assert any("memory usage may be high" in str(warning.message) for warning in w)

    def test_warns_large_memory_estimate(self):
        # num_steps=10_000, asian_tree_averages=10_000 -> ~800 GiB
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BinomialParams(num_steps=10_000, asian_tree_averages=10_000)
        assert any("GiB" in str(warning.message) for warning in w)

    @pytest.mark.parametrize(
        "field,value",
        [("num_steps", 500.0), ("mc_paths", 1000.0), ("asian_tree_averages", 100.0)],
    )
    def test_rejects_float_for_int_fields(self, field, value):
        with pytest.raises(ValidationError, match=f"{field} must be an int"):
            BinomialParams(**{field: value})

    def test_rejects_bool_for_num_steps(self):
        with pytest.raises(ValidationError, match="num_steps must be an int"):
            BinomialParams(num_steps=True)


# ---------------------------------------------------------------------------
# PDEParams
# ---------------------------------------------------------------------------


class TestPDEParams:
    def test_defaults_valid(self):
        p = PDEParams()
        assert p.smax_mult == 4.0
        assert p.method is PDEMethod.CRANK_NICOLSON

    def test_rejects_non_positive_smax_mult(self):
        with pytest.raises(ValidationError, match="smax_mult must be positive"):
            PDEParams(smax_mult=0.0)

    def test_rejects_spot_steps_below_3(self):
        with pytest.raises(ValidationError, match="spot_steps must be >= 3"):
            PDEParams(spot_steps=2)

    def test_rejects_time_steps_below_1(self):
        with pytest.raises(ValidationError, match="time_steps must be >= 1"):
            PDEParams(time_steps=0)

    def test_rejects_omega_at_boundary(self):
        with pytest.raises(ValidationError, match="omega must be in"):
            PDEParams(omega=1.0)
        with pytest.raises(ValidationError, match="omega must be in"):
            PDEParams(omega=2.0)

    def test_rejects_non_positive_tol(self):
        with pytest.raises(ValidationError, match="tol must be positive"):
            PDEParams(tol=0.0)

    def test_rejects_max_iter_below_1(self):
        with pytest.raises(ValidationError, match="max_iter must be >= 1"):
            PDEParams(max_iter=0)

    def test_rejects_negative_rannacher_steps(self):
        with pytest.raises(ValidationError, match="rannacher_steps must be >= 0"):
            PDEParams(rannacher_steps=-1)

    def test_rejects_bad_method_type(self):
        with pytest.raises(ValidationError, match="method must be a PDEMethod"):
            PDEParams(method="implicit")

    def test_rejects_bad_space_grid_type(self):
        with pytest.raises(ValidationError, match="space_grid must be a PDESpaceGrid"):
            PDEParams(space_grid="spot")

    def test_rejects_bad_american_solver_type(self):
        with pytest.raises(ValidationError, match="american_solver must be a PDEEarlyExercise"):
            PDEParams(american_solver="intrinsic")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("spot_steps", 200.0),
            ("time_steps", 200.0),
            ("max_iter", 100.0),
            ("rannacher_steps", 2.0),
        ],
    )
    def test_rejects_float_for_int_fields(self, field, value):
        with pytest.raises(ValidationError, match=f"{field} must be an int"):
            PDEParams(**{field: value})

    def test_rejects_bool_for_spot_steps(self):
        with pytest.raises(ValidationError, match="spot_steps must be an int"):
            PDEParams(spot_steps=True)

    def test_for_barriers_defaults(self):
        p = PDEParams.for_barriers()
        assert p.spot_steps == 1200
        assert p.time_steps == 800
        assert p.space_grid is PDESpaceGrid.LOG_SPOT
        assert p.method is PDEMethod.CRANK_NICOLSON
        assert p.control_variate_european is False

    def test_for_barriers_with_overrides(self):
        p = PDEParams.for_barriers(log_timings=True, control_variate_european=True)
        assert p.spot_steps == 1200
        assert p.time_steps == 800
        assert p.log_timings is True
        assert p.control_variate_european is True

    def test_for_barriers_rejects_invalid_override(self):
        with pytest.raises(ValidationError, match="spot_steps must be >= 3"):
            PDEParams.for_barriers(spot_steps=2)
