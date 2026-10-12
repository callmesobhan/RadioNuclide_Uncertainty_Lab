"""Models with several outputs: shared inputs, shared Sobol' design, joint behaviour."""

import numpy as np
import pytest

from RadioNuclide_Uncertainty_Lab import (
    InputSpace,
    Parameter,
    UncertainModel,
    budget_matrix,
    matrix_markdown,
    normal,
    propagate,
    rank_sensitivity,
    sobol_all,
    sobol_indices,
)


def _model(as_mapping=True):
    """y1 = 2a + b, y2 = b + 3c, y3 = y1 / y2 ... with a, b, c ~ N(10, 1) independent."""
    space = InputSpace([Parameter(n, normal(10, 1)) for n in ("a", "b", "c")])

    def func(x):
        y1 = 2 * x["a"] + x["b"]
        y2 = x["b"] + 3 * x["c"]
        if as_mapping:
            return {"y1": y1, "y2": y2, "ratio": y1 / y2}
        return np.column_stack([y1, y2, y1 / y2])

    return UncertainModel(
        "m", space, func, outputs=["y1", "y2", "ratio"], output_units={"ratio": ""}
    )


@pytest.mark.parametrize("as_mapping", [True, False])
def test_mapping_and_array_outputs_are_equivalent(as_mapping):
    res = propagate(_model(as_mapping), n=1024, seed=0)
    assert res.values.shape == (1024, 3)
    np.testing.assert_allclose(res["ratio"], res["y1"] / res["y2"])
    assert res.output_units == ("Gy", "Gy", "")


def test_single_output_accessors_refuse_ambiguity():
    res = propagate(_model(), n=256, seed=0)
    with pytest.raises(ValueError, match="several outputs"):
        _ = res.output
    with pytest.raises(ValueError, match="several outputs"):
        res.summary()
    with pytest.raises(KeyError, match="unknown output"):
        res.summary("nope")
    assert res.summary("y1").output == "y1"
    assert set(res.summaries()) == {"y1", "y2", "ratio"}


def test_outputs_share_inputs_so_their_correlation_is_the_analytic_one():
    # y1 = 2a + b, y2 = b + 3c: cov = Var(b) = 1; sd1 = sqrt(5), sd2 = sqrt(10)
    res = propagate(_model(), n=2**16, seed=3)
    assert res.correlation("y1", "y2") == pytest.approx(1 / np.sqrt(50), abs=0.01)


def test_sobol_all_matches_single_output_calls_and_shares_one_design():
    model = _model()
    calls = {"n": 0}
    inner = model.func

    def counting(x):
        calls["n"] += 1
        return inner(x)

    model.func = counting
    everything = sobol_all(model, n=512, seed=4, n_boot=10)
    assert calls["n"] == 2 + len(model.inputs.factors)  # one design for all outputs
    single = sobol_indices(model, n=512, seed=4, n_boot=10, output="y2")
    np.testing.assert_array_equal(single.st, everything["y2"].st)
    assert everything["y1"].output == "y1"


def test_sobol_matches_analytic_values_for_a_linear_output():
    res = sobol_all(_model(), n=2**14, seed=1)
    # y1 = 2a + b: variance shares 4/5 and 1/5, nothing for c
    np.testing.assert_allclose(res["y1"].st, [0.8, 0.2, 0.0], atol=0.02)
    # y2 = b + 3c: shares 0, 1/10, 9/10
    np.testing.assert_allclose(res["y2"].st, [0.0, 0.1, 0.9], atol=0.02)


def test_budget_matrix_orders_factors_and_renders():
    matrix = budget_matrix(sobol_all(_model(), n=2**12, seed=1))
    assert matrix.outputs == ("y1", "y2", "ratio")
    assert matrix.st.shape == (3, 3)
    assert list(matrix.st.max(axis=1)) == sorted(matrix.st.max(axis=1), reverse=True)
    md = matrix_markdown(matrix)
    assert md.startswith("| Factor | y1 | y2 | ratio |")


def test_rank_sensitivity_per_output():
    res = propagate(_model(), n=2**13, seed=2)
    r1, r2 = rank_sensitivity(res, "y1"), rank_sensitivity(res, "y2")
    assert r1.ranking("prcc")[0][0] == "a"
    assert r2.ranking("prcc")[0][0] == "c"
    with pytest.raises(ValueError, match="several outputs"):
        rank_sensitivity(res)


def test_wrong_output_shape_or_missing_key_is_rejected():
    space = InputSpace([Parameter("a", normal(1, 0.1))])
    bad_shape = UncertainModel(
        "m", space, lambda x: np.ones((len(x["a"]), 2)), outputs=["u", "v", "w"]
    )
    with pytest.raises(ValueError, match="shape"):
        propagate(bad_shape, n=8, seed=0)
    missing = UncertainModel("m", space, lambda x: {"u": x["a"]}, outputs=["u", "v"])
    with pytest.raises(ValueError, match="did not return"):
        propagate(missing, n=8, seed=0)


def test_duplicate_output_names_rejected():
    space = InputSpace([Parameter("a", normal(1, 0.1))])
    with pytest.raises(ValueError, match="duplicate output"):
        UncertainModel("m", space, lambda x: x["a"], outputs=["u", "u"])


def test_zero_variance_output_is_named_in_the_error():
    space = InputSpace([Parameter("a", normal(1, 0.1))])
    model = UncertainModel(
        "m", space, lambda x: {"u": x["a"], "const": np.ones_like(x["a"])}, outputs=["u", "const"]
    )
    with pytest.raises(ValueError, match="'const' has zero variance"):
        sobol_all(model, n=64, seed=0)
