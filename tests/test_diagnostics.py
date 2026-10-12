"""Run-time diagnostics: each rule fires on the situation it was written for, and a clean run is quiet."""

from pathlib import Path

import numpy as np

from RadioNuclide_Uncertainty_Lab import (
    InputSpace,
    Parameter,
    RunSettings,
    UncertainModel,
    analyze,
    load_config,
    lognormal,
    normal,
)

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "lu177_dotatate_multi_organ.yaml"


def codes(a):
    return {d.code for d in a.diagnostics}


def _linear(sd=(1.0, 0.3)):
    space = InputSpace(
        [Parameter("a", normal(10, sd[0]), group="A"), Parameter("b", normal(10, sd[1]), group="B")]
    )
    return UncertainModel("lin", space, lambda x: x["a"] + x["b"], outputs="y")


def test_shipped_scenario_is_clean():
    cfg = load_config(CONFIG)
    a = analyze(cfg.model, RunSettings(n=2**12, sobol_n=2**11, seed=1))
    assert [d for d in a.diagnostics if d.level == "warning"] == []


def test_missing_seed_and_small_samples_are_flagged():
    a = analyze(_linear(), RunSettings(n=256, sobol_n=256, seed=None))
    assert {"NO_SEED", "SMALL_SAMPLE", "SMALL_SOBOL"} <= codes(a)


def test_wide_sobol_confidence_intervals_are_flagged():
    a = analyze(_linear(), RunSettings(n=2**12, sobol_n=64, seed=1))
    assert "WIDE_SOBOL_CI" in codes(a)
    assert analyze(_linear(), RunSettings(n=2**12, sobol_n=2**13, seed=1)).diagnostics == ()


def test_normal_marginal_that_reaches_zero_is_flagged_and_lognormal_is_not():
    bad = InputSpace([Parameter("m", normal(1.0, 0.4))])
    good = InputSpace([Parameter("m", lognormal(1.0, 1.4))])
    run = RunSettings(n=2**12, sobol_n=2**11, seed=1)
    a_bad = analyze(
        UncertainModel("inv", bad, lambda x: 1.0 / np.maximum(x["m"], 0.02) ** 0.96, outputs="d"),
        run,
    )
    a_good = analyze(UncertainModel("inv", good, lambda x: 1.0 / x["m"] ** 0.96, outputs="d"), run)
    assert "NON_POSITIVE_SUPPORT" in codes(a_bad)
    assert "NON_POSITIVE_SUPPORT" not in codes(a_good)
    # the hazard it warns about is real: heavy-tailed output
    assert "HEAVY_TAIL" in codes(a_bad)


def test_no_sobol_is_reported_as_information_only():
    a = analyze(_linear(), RunSettings(n=2**12, seed=1), sobol=False)
    assert ("info", "NO_SOBOL") in {(d.level, d.code) for d in a.diagnostics}
    assert a.budgets is None and a.matrix is None


def test_strong_interaction_and_non_linearity_are_reported_as_information():
    space = InputSpace([Parameter(n, lognormal(1.0, 3.0), group=n) for n in ("a", "b")])
    a = analyze(
        UncertainModel("prod", space, lambda x: x["a"] * x["b"], outputs="y"),
        RunSettings(n=2**13, sobol_n=2**13, seed=1),
    )
    assert {"NON_LINEAR", "INTERACTIONS"} <= codes(a)
    assert all(d.level in ("info", "warning") for d in a.diagnostics)


def test_diagnostics_are_ordered_warnings_first():
    a = analyze(_linear(), RunSettings(n=256, sobol_n=64, seed=None), sobol=True)
    levels = [d.level for d in a.diagnostics]
    assert levels == sorted(levels, key=lambda lv: lv != "warning")


def test_diagnostics_never_change_the_numbers():
    run = RunSettings(n=2**12, sobol_n=2**11, seed=5)
    a = analyze(_linear(), run)
    b = analyze(_linear(), run)
    np.testing.assert_array_equal(a.result.values, b.result.values)
    assert a.diagnostics == b.diagnostics
