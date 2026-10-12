import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from RadioNuclide_Uncertainty_Lab import (
    InputSpace,
    Parameter,
    lognormal,
    normal,
    triangular,
    truncated_normal,
    uniform,
)
from RadioNuclide_Uncertainty_Lab.distributions import from_spec
from RadioNuclide_Uncertainty_Lab.sampling import unit_samples


def test_lognormal_median_and_gsd():
    d = lognormal(median=2.0, gsd=1.3)
    x = d.rvs(200_000, random_state=1)
    assert np.median(x) == pytest.approx(2.0, rel=0.01)
    assert math.exp(np.std(np.log(x))) == pytest.approx(1.3, rel=0.005)


def test_normal_rel_sd_and_truncation():
    assert normal(10, rel_sd=0.1).std() == pytest.approx(1.0)
    x = truncated_normal(0.5, 0.5, low=0.0, high=1.0).rvs(10_000, random_state=0)
    assert x.min() >= 0 and x.max() <= 1


@pytest.mark.parametrize(
    "build",
    [
        lambda: lognormal(1, 1.0),
        lambda: lognormal(-1, 1.2),
        lambda: normal(1),
        lambda: normal(1, 0.1, rel_sd=0.1),
        lambda: uniform(2, 1),
        lambda: triangular(0, 2, 1),
    ],
)
def test_bad_distribution_arguments_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_from_spec_rejects_typos_and_missing_keys():
    assert from_spec({"dist": "uniform", "low": 0, "high": 1}).mean() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="does not accept"):
        from_spec({"dist": "lognormal", "median": 1, "gsd": 1.2, "sd": 3})
    with pytest.raises(ValueError, match="missing required key"):
        from_spec({"dist": "lognormal", "median": 1})
    with pytest.raises(ValueError, match="unknown dist"):
        from_spec({"dist": "weibul"})


def test_sobol_sampling_rounds_up_to_power_of_two():
    assert len(unit_samples(1000, 3, "sobol", seed=0)) == 1024
    assert len(unit_samples(1000, 3, "lhs", seed=0)) == 1000


def test_same_seed_same_samples():
    np.testing.assert_array_equal(unit_samples(64, 4, seed=7), unit_samples(64, 4, seed=7))


def _p(name, group=None):
    return Parameter(name, normal(0, 1), group=group)


def test_factors_group_parameters_and_keep_ungrouped_separate():
    space = InputSpace([_p("a", "g"), _p("b"), _p("c", "g")])
    assert [(f.name, f.members) for f in space.factors] == [("g", ("a", "c")), ("b", ("b",))]


@pytest.mark.parametrize(
    "params, corr, message",
    [
        ([_p("a"), _p("a")], None, "duplicate"),
        ([_p("a", "g"), _p("b", "h")], {("a", "b"): 0.5}, "share a group"),
        ([_p("a"), _p("b")], {("a", "b"): 0.5}, "share a group"),
        ([_p("a", "g"), _p("b", "g")], {("a", "b"): 1.0}, r"\(-1, 1\)"),
        ([_p("a", "g"), _p("b", "g")], {("a", "z"): 0.1}, "unknown parameter"),
        (
            [_p("a", "g"), _p("b", "g"), _p("c", "g")],
            {("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "c"): -0.9},
            "positive definite",
        ),
    ],
)
def test_invalid_input_spaces_rejected(params, corr, message):
    with pytest.raises(ValueError, match=message):
        InputSpace(params, corr)


def test_gaussian_copula_induces_requested_dependence():
    rho = 0.7
    space = InputSpace([_p("a", "g"), _p("b", "g"), _p("c")], {("a", "b"): rho})
    x = space.transform(unit_samples(2**15, 3, "sobol", seed=3))
    # Gaussian copula: Spearman = 6/pi * asin(rho/2); normal marginals => Pearson = rho
    assert spearmanr(x["a"], x["b"])[0] == pytest.approx(6 / math.pi * math.asin(rho / 2), abs=0.02)
    assert np.corrcoef(x["a"], x["b"])[0, 1] == pytest.approx(rho, abs=0.02)
    assert abs(spearmanr(x["a"], x["c"])[0]) < 0.03  # other factors stay independent
