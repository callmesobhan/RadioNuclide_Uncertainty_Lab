"""The multi-organ MIRD model: arithmetic, scaling, cross-dose, shared vs specific factors."""

import numpy as np
import pytest

from RadioNuclide_Uncertainty_Lab import (
    Parameter,
    multi_organ,
    normal,
    propagate,
    sobol_all,
)
from RadioNuclide_Uncertainty_Lab.kinetics import LN2

ORGANS = {
    "A": {"s_self_gy_per_mbq_h": 2e-4, "mass_ref_kg": 0.4, "kinetics": "tia_coefficient"},
    "B": {
        "s_self_gy_per_mbq_h": 4e-5,
        "mass_ref_kg": 2.0,
        "kinetics": "mono_exponential",
    },
}
FIXED = {
    "administered_activity_mbq": 1000.0,
    "A.tia_coefficient_h": 3.0,
    "B.uptake_fraction": 0.05,
    "B.effective_half_life_h": 60.0,
}


def _build(parameters=(), fixed=None, **kw):
    return multi_organ(list(parameters), fixed={**FIXED, **(fixed or {})}, organs=ORGANS, **kw)


def _eval(model, **x):
    out = model.func({k: np.atleast_1d(np.asarray(v, float)) for k, v in x.items()})
    return {k: float(v[0]) for k, v in out.items()}


def _with_dummy(model_builder, **kw):
    # the input space needs at least one uncertain parameter; add an inert scale factor
    dummy = Parameter("activity_scale.dummy", normal(1.0, 0.01))
    return model_builder([dummy], **kw), {"activity_scale.dummy": 1.0}


def test_hand_calculation_for_both_kinetics_types():
    model, x = _with_dummy(_build)
    d = _eval(model, **x)
    tia_a = 3.0 * 1000.0
    tia_b = 0.05 * 1000.0 * 60.0 / LN2
    assert d["A"] == pytest.approx(tia_a * 2e-4, rel=1e-12)
    assert d["B"] == pytest.approx(tia_b * 4e-5, rel=1e-12)


def test_self_s_value_scales_with_mass_to_the_minus_exponent():
    mass = Parameter("A.mass_kg", normal(0.4, 0.04))
    model = _build([mass])
    base = _eval(model, **{"A.mass_kg": 0.4})["A"]
    assert _eval(model, **{"A.mass_kg": 0.8})["A"] == pytest.approx(base / 2)
    organs = {**ORGANS, "A": {**ORGANS["A"], "mass_exponent": 0.961}}
    steep = multi_organ([mass], fixed=FIXED, organs=organs)
    assert _eval(steep, **{"A.mass_kg": 0.8})["A"] == pytest.approx(base * 2**-0.961)


def test_cross_dose_adds_the_source_tia_times_cross_s_value():
    model, x = _with_dummy(_build, cross_s_gy_per_mbq_h={"A": {"B": 5e-6}, "B": {"A": 1e-6}})
    d = _eval(model, **x)
    tia_a = 3000.0
    tia_b = 0.05 * 1000.0 * 60.0 / LN2
    assert d["A"] == pytest.approx(tia_a * 2e-4 + tia_b * 5e-6, rel=1e-12)
    assert d["B"] == pytest.approx(tia_b * 4e-5 + tia_a * 1e-6, rel=1e-12)


def test_shared_activity_scale_hits_every_organ_and_specific_ones_only_their_own():
    params = [
        Parameter("activity_scale.shared", normal(1.0, 0.1)),
        Parameter("A.activity_scale.own", normal(1.0, 0.1)),
        Parameter("B.dose_scale.own", normal(1.0, 0.1)),
    ]
    model = _build(params)
    base = _eval(
        model, **{"activity_scale.shared": 1, "A.activity_scale.own": 1, "B.dose_scale.own": 1}
    )
    d = _eval(
        model, **{"activity_scale.shared": 1.1, "A.activity_scale.own": 1, "B.dose_scale.own": 1}
    )
    assert d["A"] == pytest.approx(1.1 * base["A"]) and d["B"] == pytest.approx(1.1 * base["B"])
    d = _eval(
        model, **{"activity_scale.shared": 1, "A.activity_scale.own": 1.2, "B.dose_scale.own": 1}
    )
    assert d["A"] == pytest.approx(1.2 * base["A"]) and d["B"] == pytest.approx(base["B"])
    d = _eval(
        model, **{"activity_scale.shared": 1, "A.activity_scale.own": 1, "B.dose_scale.own": 0.9}
    )
    assert d["A"] == pytest.approx(base["A"]) and d["B"] == pytest.approx(0.9 * base["B"])


def test_shared_factors_cancel_exactly_in_ratios():
    shared = Parameter("activity_scale.shared", normal(1.0, 0.2), group="calibration")
    model = _build([shared], ratios=[{"name": "A_to_B", "numerator": "A", "denominator": "B"}])
    res = propagate(model, n=1024, seed=0)
    assert np.ptp(res["A_to_B"]) / res["A_to_B"].mean() < 1e-12  # exactly constant
    assert res["A"].std() / res["A"].mean() == pytest.approx(0.2, rel=0.05)
    assert model.unit("A_to_B") == "" and model.unit("A") == "Gy"


def test_shared_error_induces_output_correlation_and_specific_errors_dilute_it():
    params = [
        Parameter("activity_scale.shared", normal(1.0, 0.10), group="calibration"),
        Parameter("A.activity_scale.own", normal(1.0, 0.10), group="a_specific"),
        Parameter("B.activity_scale.own", normal(1.0, 0.10), group="b_specific"),
    ]
    res = propagate(_build(params), n=2**15, seed=1)
    # equal variances 0.1^2 each: corr = shared / (shared + own) = 1/2 (to first order)
    assert res.correlation("A", "B") == pytest.approx(0.5, abs=0.03)
    # organ B's own error cannot touch organ A's dose
    sob = sobol_all(_build(params), n=2**13, seed=1)
    st_a = dict(zip(sob["A"].factors, sob["A"].st, strict=True))
    assert st_a["b_specific"] == pytest.approx(0.0, abs=0.02)
    assert st_a["calibration"] == pytest.approx(0.5, abs=0.05)


def test_validation_errors():
    dummy = [Parameter("activity_scale.dummy", normal(1.0, 0.01))]
    no_activity = {k: v for k, v in FIXED.items() if k != "administered_activity_mbq"}
    with pytest.raises(ValueError, match="administered_activity_mbq"):
        multi_organ(dummy, fixed=no_activity, organs=ORGANS)
    missing = {k: v for k, v in FIXED.items() if k != "B.effective_half_life_h"}
    with pytest.raises(ValueError, match="missing quantities.*B.effective_half_life_h"):
        multi_organ(dummy, fixed=missing, organs=ORGANS)
    with pytest.raises(ValueError, match="both uncertain and fixed"):
        multi_organ([Parameter("A.tia_coefficient_h", normal(3, 0.3))], fixed=FIXED, organs=ORGANS)
    with pytest.raises(ValueError, match="unknown quantities.*C.activity_scale.x"):
        multi_organ(
            [Parameter("C.activity_scale.x", normal(1, 0.1))], fixed=FIXED, organs=ORGANS
        )  # organ typo
    with pytest.raises(ValueError, match="unknown quantities.*A.dose_scale"):
        multi_organ([Parameter("A.dose_scale", normal(1, 0.1))], fixed=FIXED, organs=ORGANS)
    with pytest.raises(ValueError, match="kinetics must be one of"):
        multi_organ(dummy, fixed=FIXED, organs={"A": {**ORGANS["A"], "kinetics": "magic"}})
    with pytest.raises(ValueError, match="unknown keys"):
        multi_organ(dummy, fixed=FIXED, organs={"A": {**ORGANS["A"], "mas_kg": 1}})
    with pytest.raises(ValueError, match="invalid organ name"):
        multi_organ(dummy, fixed=FIXED, organs={"a.b": ORGANS["A"]})
    with pytest.raises(ValueError, match="self term"):
        _build(dummy, cross_s_gy_per_mbq_h={"A": {"A": 1e-5}})
    with pytest.raises(ValueError, match="unknown source organ"):
        _build(dummy, cross_s_gy_per_mbq_h={"A": {"Z": 1e-5}})
    with pytest.raises(ValueError, match="clashes with an organ"):
        _build(dummy, ratios=[{"name": "A", "numerator": "A", "denominator": "B"}])
    with pytest.raises(ValueError, match="unknown organ"):
        _build(dummy, ratios=[{"name": "r", "numerator": "A", "denominator": "Q"}])
