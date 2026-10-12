"""Propagation, sensitivity and error-budget tests against closed-form answers."""

import math

import numpy as np
import pytest

from RadioNuclide_Uncertainty_Lab import (
    InputSpace,
    Parameter,
    UncertainModel,
    budget_markdown,
    error_budget,
    lognormal,
    normal,
    propagate,
    rank_sensitivity,
    sobol_indices,
    uniform,
)

# --- propagation ---------------------------------------------------------------------------


def _product_model(medians=(1.0, 2.0, 3.0), gsds=(1.1, 1.3, 1.5), groups=("a", "b", "c")):
    params = [
        Parameter(f"x{i}", lognormal(m, g), group=grp)
        for i, (m, g, grp) in enumerate(zip(medians, gsds, groups, strict=True), 1)
    ]
    return UncertainModel(
        "product", InputSpace(params), lambda x: x["x1"] * x["x2"] * x["x3"], "y", ""
    )


def test_multiplicative_lognormal_propagation_matches_theory():
    gsds = (1.1, 1.3, 1.5)
    res = propagate(_product_model(gsds=gsds), n=2**16, seed=1)
    s = res.summary()
    assert s.median == pytest.approx(6.0, rel=0.01)
    # log-variances add for a product of independent lognormals
    assert s.gsd == pytest.approx(
        math.exp(math.sqrt(sum(math.log(g) ** 2 for g in gsds))), rel=0.01
    )
    assert s.ci_low < s.median < s.ci_high
    assert s.skewness > 0


def test_propagation_is_reproducible_and_seed_dependent():
    m = _product_model()
    a, b, c = (propagate(m, n=512, seed=s).output for s in (5, 5, 6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_propagate_rounds_sobol_up_and_lhs_exact():
    m = _product_model()
    assert propagate(m, n=1000, method="sobol", seed=0).n == 1024
    assert propagate(m, n=1000, method="lhs", seed=0).n == 1000


# --- rank-based screening --------------------------------------------------------------------


def test_rank_sensitivity_identifies_direction_and_irrelevant_input():
    space = InputSpace([Parameter(n, normal(0, 1)) for n in ("a", "b", "noise")])
    model = UncertainModel("lin", space, lambda x: 3 * x["a"] - 1 * x["b"])
    r = rank_sensitivity(propagate(model, n=2**13, seed=2))
    ranking = dict(r.ranking("prcc"))
    assert ranking["a"] > 0.9 and ranking["b"] < -0.5
    assert abs(ranking["noise"]) < 0.05
    assert [n for n, _ in r.ranking("prcc")][:2] == ["a", "b"]
    assert r.r2 == pytest.approx(1.0, abs=1e-6)
    # SRC of a linear model: coefficient * sd_x / sd_y
    sd_y = math.sqrt(9 + 1)
    assert r.src[0] == pytest.approx(3 / sd_y, abs=0.01)


# --- Sobol' indices ----------------------------------------------------------------------------


def test_sobol_matches_ishigami_analytic_values():
    space = InputSpace([Parameter(f"x{i}", uniform(-math.pi, math.pi)) for i in (1, 2, 3)])

    def ishigami(x):
        return np.sin(x["x1"]) + 7 * np.sin(x["x2"]) ** 2 + 0.1 * x["x3"] ** 4 * np.sin(x["x1"])

    res = sobol_indices(UncertainModel("ishigami", space, ishigami), n=2**14, seed=0)
    np.testing.assert_allclose(res.s1, [0.3139, 0.4424, 0.0], atol=0.03)
    np.testing.assert_allclose(res.st, [0.5576, 0.4424, 0.2437], atol=0.03)
    assert res.n_evaluations == 2**14 * 5


def test_sobol_matches_analytic_values_for_multiplicative_lognormal_model():
    medians, gsds = (1.0, 2.0, 3.0), (1.1, 1.3, 1.5)
    res = sobol_indices(_product_model(medians, gsds), n=2**15, seed=3)
    # Y = prod X_i, independent lognormals: mean m_i, variance v_i
    s2 = [math.log(g) ** 2 for g in gsds]
    m2 = [(med * math.exp(s / 2)) ** 2 for med, s in zip(medians, s2, strict=True)]
    v = [(math.exp(s) - 1) * mm for s, mm in zip(s2, m2, strict=True)]
    total_var = math.prod(a + b for a, b in zip(v, m2, strict=True)) - math.prod(m2)
    s1 = [v[i] * math.prod(m2[j] for j in range(3) if j != i) / total_var for i in range(3)]
    st = [v[i] * math.prod(v[j] + m2[j] for j in range(3) if j != i) / total_var for i in range(3)]
    np.testing.assert_allclose(res.s1, s1, atol=0.03)
    np.testing.assert_allclose(res.st, st, atol=0.03)
    assert res.interaction_share > 0  # products interact; the budget must not hide it


def test_sobol_treats_a_correlated_group_as_one_factor():
    a, b, c = 2.0, 1.0, 1.5
    rho = 0.5
    space = InputSpace(
        [
            Parameter("x1", normal(0, 1), group="pair"),
            Parameter("x2", normal(0, 1), group="pair"),
            Parameter("x3", normal(0, 1)),
        ],
        {("x1", "x2"): rho},
    )
    model = UncertainModel("additive", space, lambda x: a * x["x1"] + b * x["x2"] + c * x["x3"])
    res = sobol_indices(model, n=2**14, seed=4)
    var_pair = a**2 + b**2 + 2 * a * b * rho
    total = var_pair + c**2
    assert res.factors == ("pair", "x3")
    assert res.members[0] == ("x1", "x2")
    np.testing.assert_allclose(res.s1, [var_pair / total, c**2 / total], atol=0.02)
    np.testing.assert_allclose(res.st, res.s1, atol=0.02)  # additive => no interactions
    assert res.interaction_share == pytest.approx(0.0, abs=0.02)


def test_sobol_is_reproducible():
    m = _product_model()
    a = sobol_indices(m, n=256, seed=9, n_boot=20)
    b = sobol_indices(m, n=256, seed=9, n_boot=20)
    np.testing.assert_array_equal(a.s1, b.s1)
    np.testing.assert_array_equal(a.st_ci, b.st_ci)


def test_sobol_confidence_intervals_bracket_estimates_and_shrink_with_n():
    m = _product_model()
    small = sobol_indices(m, n=2**9, seed=0)
    large = sobol_indices(m, n=2**13, seed=0)
    for res in (small, large):
        assert np.all(res.s1_ci[:, 0] <= res.s1_ci[:, 1])
        assert np.all((res.st_ci[:, 0] <= res.st + 0.05) & (res.st <= res.st_ci[:, 1] + 0.05))
    assert np.mean(large.st_ci[:, 1] - large.st_ci[:, 0]) < np.mean(
        small.st_ci[:, 1] - small.st_ci[:, 0]
    )


def test_constant_output_is_rejected():
    space = InputSpace([Parameter("x", normal(0, 1))])
    model = UncertainModel("const", space, lambda x: np.ones_like(x["x"]))
    with pytest.raises(ValueError, match="zero variance"):
        sobol_indices(model, n=64, seed=0)


# --- error budget ----------------------------------------------------------------------------


def test_error_budget_sorts_by_total_effect_and_converts_to_sd():
    model = _product_model()
    sobol = sobol_indices(model, n=2**12, seed=1)
    budget = error_budget(sobol, "Gy")
    assert [ln.st for ln in budget.lines] == sorted((ln.st for ln in budget.lines), reverse=True)
    assert budget.lines[0].factor == "c"  # largest gsd (1.5)
    for ln in budget.lines:
        assert ln.sd_first == pytest.approx(math.sqrt(max(ln.s1, 0) * sobol.variance))
        assert ln.rel_total == pytest.approx(ln.sd_total / sobol.mean)
    assert budget.relative_sd == pytest.approx(math.sqrt(sobol.variance) / sobol.mean)
    md = budget_markdown(budget)
    assert md.count("\n") >= len(budget.lines) + 2
    assert "interactions" in md
