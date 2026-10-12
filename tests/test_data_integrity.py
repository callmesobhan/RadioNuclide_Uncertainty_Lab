"""Guarantees about the shipped data: derived values are derived, nothing drifts, cross-dose is live."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from RadioNuclide_Uncertainty_Lab import datacheck as dc
from RadioNuclide_Uncertainty_Lab import get_registry, load_config, load_svalues, propagate
from RadioNuclide_Uncertainty_Lab.mird import tia_coefficients_from_doses

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "lu177_dotatate_multi_organ.yaml"
REGIONS = {"kidneys": "Kidneys", "liver": "Liver", "spleen": "Spleen"}


def _raw():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def derived():
    reg, ds = get_registry(), load_svalues("lu177_adult_male")
    s = np.array([[ds.s(REGIONS[t], REGIONS[src]) for src in REGIONS] for t in REGIONS])
    doses = np.array(
        [
            reg.value("staanum2021.dotatate.specific_dose.mean"),
            reg.value("santoro2018.dose.liver"),
            reg.value("santoro2018.dose.spleen"),
        ]
    )
    tia = tia_coefficients_from_doses(s, doses)
    teff_h = reg.value("staanum2021.dotatate.teff.mean") * 24.0
    return {
        "s": s,
        "doses": doses,
        "tia": tia,
        "uptake": tia[0] * np.log(2) / teff_h,
        "teff_h": teff_h,
    }


def test_config_anchors_are_the_derived_values_not_hand_typed_guesses(derived):
    raw = _raw()
    assert raw["fixed"]["liver.tia_coefficient_h"] == pytest.approx(derived["tia"][1], rel=5e-4)
    assert raw["fixed"]["spleen.tia_coefficient_h"] == pytest.approx(derived["tia"][2], rel=5e-4)
    assert raw["parameters"]["kidneys.uptake_fraction"]["median"] == pytest.approx(
        derived["uptake"], rel=5e-4
    )
    assert raw["parameters"]["kidneys.effective_half_life_h"]["median"] == pytest.approx(
        derived["teff_h"], rel=1e-9
    )
    assert raw["fixed"]["administered_activity_mbq"] == get_registry().value(
        "staanum2021.administered_activity"
    )


def test_every_uncertain_parameter_of_the_shipped_scenario_cites_a_claim():
    cfg = load_config(CONFIG)
    uncited = [p.name for p in cfg.model.inputs.parameters if not p.claim]
    assert uncited == []
    reg = get_registry()
    assert all(c in reg.claims and c not in reg.excluded for c in cfg.claims_used)
    assert set(cfg.data_integrity()["excluded_claims"]) == set(reg.excluded)


def test_cross_dose_is_about_two_percent_of_the_kidney_dose_as_the_eanm_text_says(derived):
    s, tia = derived["s"], derived["tia"]
    from_others = s[0, 1] * tia[1] + s[0, 2] * tia[2]
    share = 100 * from_others / (s[0] @ tia)
    assert share == pytest.approx(
        get_registry().value("gleisner2022.rest_of_body_share_kidney_dose"), abs=0.7
    )
    # and it is small for liver and spleen, whose self-dose is much larger
    assert 100 * (s[1, 0] * tia[0] + s[1, 2] * tia[2]) / (s[1] @ tia) < 0.5
    assert 100 * (s[2, 0] * tia[0] + s[2, 1] * tia[1]) / (s[2] @ tia) < 1.0


def test_cross_dose_switch_is_live_and_changes_doses_by_the_expected_amount():
    medians = lambda cfg: {  # noqa: E731
        r["name"]: np.array([r["median"]]) for r in cfg.model.inputs.describe()
    }
    with_cross = load_config(CONFIG)
    raw = _raw()
    raw["cross_dose"] = False
    without = load_config(raw)
    a = with_cross.model.func(medians(with_cross))
    b = without.model.func(medians(without))
    # without cross terms the dose falls by exactly the cross share
    assert 1 - b["kidneys"][0] / a["kidneys"][0] == pytest.approx(0.018, abs=0.004)
    assert 1 - b["liver"][0] / a["liver"][0] == pytest.approx(0.0025, abs=0.002)
    assert with_cross.model.metadata["cross_s_gy_per_mbq_h"]["kidneys"]["liver"] > 0
    assert without.model.metadata["cross_s_gy_per_mbq_h"] == {}


def test_cross_dose_does_not_change_the_uncertainty_budget_materially():
    a = propagate(load_config(CONFIG).model, n=2**14, seed=3).summary("kidneys").cv
    raw = _raw()
    raw["cross_dose"] = False
    b = propagate(load_config(raw).model, n=2**14, seed=3).summary("kidneys").cv
    assert a == pytest.approx(b, rel=0.03)


def test_explicit_and_dataset_cross_terms_may_not_both_define_a_pair():
    raw = _raw()
    raw["cross_s_gy_per_mbq_h"] = {"kidneys": {"liver": 1e-6}}
    with pytest.raises(ValueError, match="both explicitly and by the dataset"):
        load_config(raw)


def test_model_refuses_a_dataset_whose_checks_fail(monkeypatch):
    from importlib import resources

    raw = yaml.safe_load(
        (resources.files("RadioNuclide_Uncertainty_Lab") / "data" / "claims.yaml").read_text()
    )
    raw["checks"] = [c for c in raw["checks"] if c["kind"] != "refdata"]  # dataset loses its checks
    no_dataset_checks = dc.load_registry(yaml.safe_dump(raw))
    monkeypatch.setattr(dc, "get_registry", lambda: no_dataset_checks)
    with pytest.raises(ValueError, match="no registered consistency checks"):
        load_config(CONFIG)

    raw["checks"].append(
        {
            "id": "svalues.always_fails",
            "kind": "bound",
            "value": 2,
            "lo": 5,
            "hi": 6,
            "dataset": "lu177_adult_male",
            "expect": "pass",
        }
    )
    failing = dc.load_registry(yaml.safe_dump(raw))
    monkeypatch.setattr(dc, "get_registry", lambda: failing)
    with pytest.raises(ValueError, match="failed consistency checks.*always_fails"):
        load_config(CONFIG)


def test_anchoring_helper_validates_its_input():
    s = np.array([[2e-4, 5e-7], [5e-7, 4e-5]])
    t = tia_coefficients_from_doses(s, np.array([0.46, 0.54]))
    np.testing.assert_allclose(s @ t * 1000, [0.46, 0.54])
    with pytest.raises(ValueError, match="square"):
        tia_coefficients_from_doses(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="inconsistent"):
        tia_coefficients_from_doses(s, np.array([0.46, -0.1]))
    with pytest.raises(np.linalg.LinAlgError):
        tia_coefficients_from_doses(np.ones((2, 2)), np.ones(2))


def test_scenario_nominal_total_doses_still_equal_the_published_means_with_cross_dose():
    cfg = copy.deepcopy(load_config(CONFIG))
    medians = {r["name"]: np.array([r["median"]]) for r in cfg.model.inputs.describe()}
    dose = cfg.model.func(medians)
    gbq = 7.4
    assert dose["kidneys"][0] / gbq == pytest.approx(0.46, rel=2e-3)
    assert dose["liver"][0] / gbq == pytest.approx(0.54, rel=2e-3)
    assert dose["spleen"][0] / gbq == pytest.approx(0.61, rel=2e-3)
