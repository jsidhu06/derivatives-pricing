"""External American benchmark: Longstaff & Schwartz (2001).

"Valuing American Options by Simulation: A Simple Least-Squares Approach",
Review of Financial Studies 14(1), Table 1 (p. 127). American put, K = 40,
r = 0.06, exercisable 50 times per year. The paper's finite-difference column
is therefore a Bermudan price (50 exercise dates a year), not continuous
exercise.
"""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import ExerciseType, OptionType, PDESpaceGrid, PricingMethod
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from derivatives_pricing.valuation import (
    BinomialParams,
    MonteCarloParams,
    OptionValuation,
    PDEParams,
    UnderlyingData,
    VanillaSpec,
)
from helpers import PRICING_DATE, flat_curve

_STRIKE = 40.0
_RATE = 0.06
_ROUNDING = 0.0005  # half a unit in the table's third decimal place

# (S, sigma, T, finite-difference Bermudan, closed-form European, LSM, LSM s.e.)
_TABLE_1 = [
    (36, 0.20, 1, 4.478, 3.844, 4.472, 0.010),
    (36, 0.20, 2, 4.840, 3.763, 4.821, 0.012),
    (36, 0.40, 1, 7.101, 6.711, 7.091, 0.020),
    (36, 0.40, 2, 8.508, 7.700, 8.488, 0.024),
    (38, 0.20, 1, 3.250, 2.852, 3.244, 0.009),
    (38, 0.20, 2, 3.745, 2.991, 3.735, 0.011),
    (38, 0.40, 1, 6.148, 5.834, 6.139, 0.019),
    (38, 0.40, 2, 7.670, 6.979, 7.669, 0.022),
    (40, 0.20, 1, 2.314, 2.066, 2.313, 0.009),
    (40, 0.20, 2, 2.885, 2.356, 2.879, 0.010),
    (40, 0.40, 1, 5.312, 5.060, 5.308, 0.018),
    (40, 0.40, 2, 6.920, 6.326, 6.921, 0.022),
    (42, 0.20, 1, 1.617, 1.465, 1.617, 0.007),
    (42, 0.20, 2, 2.212, 1.841, 2.206, 0.010),
    (42, 0.40, 1, 4.582, 4.379, 4.588, 0.017),
    (42, 0.40, 2, 6.248, 5.736, 6.243, 0.021),
    (44, 0.20, 1, 1.110, 1.017, 1.118, 0.007),
    (44, 0.20, 2, 1.690, 1.429, 1.675, 0.009),
    (44, 0.40, 1, 3.948, 3.783, 3.957, 0.017),
    (44, 0.40, 2, 5.647, 5.202, 5.622, 0.021),
]
_IDS = [f"S{s}-vol{v}-T{t}" for s, v, t, *_ in _TABLE_1]


def _maturity(years: int) -> dt.datetime:
    return dt.datetime(2025 + years, 1, 1)  # T = 1 or 2 years (ACT/365F)


def _market(years: int) -> MarketData:
    return MarketData(
        PRICING_DATE, flat_curve(PRICING_DATE, _maturity(years), _RATE), currency="USD"
    )


def _put(years: int, exercise: ExerciseType) -> VanillaSpec:
    return VanillaSpec(
        option_type=OptionType.PUT,
        exercise_type=exercise,
        strike=_STRIKE,
        maturity=_maturity(years),
    )


def _underlying(spot: float, vol: float, years: int) -> UnderlyingData:
    return UnderlyingData(initial_value=spot, volatility=vol, market_data=_market(years))


@pytest.mark.parametrize("spot,vol,years,fd,european,lsm,lsm_se", _TABLE_1, ids=_IDS)
def test_closed_form_european_column(spot, vol, years, fd, european, lsm, lsm_se):
    pv = OptionValuation(
        _underlying(spot, vol, years),
        _put(years, ExerciseType.EUROPEAN),
        PricingMethod.ANALYTICAL,
    ).present_value()
    assert abs(pv - european) <= _ROUNDING + 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("spot,vol,years,fd,european,lsm,lsm_se", _TABLE_1, ids=_IDS)
def test_tree_and_pde_american_bound_the_bermudan_column(
    spot, vol, years, fd, european, lsm, lsm_se
):
    """The tree and PDE exercise continuously, so they must not price below the
    paper's 50-dates-a-year Bermudan value (allowing for its rounding and our
    discretisation error). The excess is the value of exercising between those
    dates: measured up to 0.009, and within the table's rounding for S = 44,
    sigma = 0.4, T = 2. The 0.015 cap catches overpricing. The PDE uses the
    log-spot grid: the uniform grid's capped domain sits only ~2.3 standard
    deviations out at sigma = 0.4, T = 2."""
    discretisation, max_excess = 0.0005, 0.015
    ud = _underlying(spot, vol, years)
    spec = _put(years, ExerciseType.AMERICAN)
    tree = OptionValuation(
        ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=5000)
    ).present_value()
    pde = OptionValuation(
        ud,
        spec,
        PricingMethod.PDE_FD,
        params=PDEParams(spot_steps=800, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT),
    ).present_value()
    for american in (tree, pde):
        assert fd - _ROUNDING - discretisation < american < fd + max_excess, (american, fd)


@pytest.mark.slow
def test_lsm_matches_the_bermudan_column():
    """Same setting as the paper: 100,000 paths (50,000 plus antithetic) and 50
    exercise dates a year, so LSM prices exactly the Bermudan option in the
    finite-difference column, which has no sampling error. The regression
    differs: the paper uses a constant plus three exponentially weighted
    Laguerre polynomials of S; we use Laguerre polynomials of S/K with a small
    ridge penalty. For a one-factor put both bases give nearly the same
    exercise policy, and the 0.005 allowance covers the difference.

    Each row alone allows 2-4 cents. Each row gets its own seed, so the 20
    sampling errors are independent and the mean difference has standard error
    sqrt(sum se_i^2) / 20 (~0.002): the aggregate check fails a systematic bias
    of about a cent. (With one shared seed, rows with the same T would reuse the
    same shocks and their errors would correlate.)"""
    basis_allowance = 0.005
    diffs, std_errors = [], []
    for seed, (spot, vol, years, fd, *_) in enumerate(_TABLE_1, start=1):
        process = GBMProcess(
            _market(years),
            GBMParams(initial_value=float(spot), volatility=vol),
            SimulationConfig(
                paths=100_000,
                num_steps=50 * years,
                end_date=_maturity(years),
                antithetic=True,
                moment_matching=False,
            ),
        )
        valuation = OptionValuation(
            process,
            _put(years, ExerciseType.AMERICAN),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=seed),
        )
        se = valuation.std_error()
        assert se is not None
        diff = valuation.present_value() - fd
        assert abs(diff) < 3.0 * se + basis_allowance + _ROUNDING, (spot, vol, years, diff, se)
        diffs.append(diff)
        std_errors.append(se)

    mean_diff = float(np.mean(diffs))
    se_of_mean = float(np.sqrt(np.sum(np.square(std_errors)))) / len(diffs)
    assert abs(mean_diff) < 3.0 * se_of_mean + basis_allowance, (mean_diff, se_of_mean)
