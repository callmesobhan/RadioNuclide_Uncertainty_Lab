"""Independent second sources for the cross-organ S-values and the nuclear data, and proof that the
checks built on them can fail."""

import copy
import dataclasses
import json
from importlib import resources

import pytest
import yaml

from RadioNuclide_Uncertainty_Lab import datacheck as dc
from RadioNuclide_Uncertainty_Lab import load_svalues, refdata
from RadioNuclide_Uncertainty_Lab import nuclides as nuclides_mod
from RadioNuclide_Uncertainty_Lab.datacheck import (
    FAIL,
    KNOWN_CONFLICT,
    PASS,
    UNEXPECTED_PASS,
    get_registry,
    load_registry,
)

DATA = resources.files("RadioNuclide_Uncertainty_Lab") / "data"


def _status(registry):
    return {r.id: r for r in registry.results}


def _claims_text():
    return (DATA / "claims.yaml").read_text()


def _with_dataset(monkeypatch, name, mutate):
    raw = json.loads((DATA / f"{name}.json").read_text())
    broken = copy.deepcopy(raw)
    mutate(broken)
    monkeypatch.setattr(refdata, "cached_svalues", lambda n: refdata._validate_and_build(broken, n))
    return load_registry()


# --- the female phantom as an independent test of the cross terms -----------------------------


def test_female_dataset_is_packaged_with_its_own_provenance():
    ds = load_svalues("lu177_adult_female")
    assert ds.phantom == "ICRP Adult Female"
    assert {
        "Kidneys",
        "Liver",
        "Spleen",
        "Pancreas",
        "Gallbladder wall",
        "Brain",
        "Thyroid",
    } <= set(ds.regions)
    prov = json.loads((DATA / "lu177_adult_female.json").read_text())["provenance"]
    assert prov["member_crc32"] == "8561c4f6"
    assert prov["archive_member"].endswith("SDATA_Lu-177_ICRP Adult Female.csv")
    assert prov["bytes_fetched"] < 5_000_000
    assert load_svalues("lu177_adult_male").fingerprint != ds.fingerprint


def test_mirdcalc_cross_terms_agree_with_idac_and_opendose_to_half_a_percent():
    st = _status(get_registry())
    assert st["svalues.female.cross_vs_idac"].status == PASS
    assert st["svalues.female.cross_vs_opendose"].status == PASS
    assert "6 pairs" in st["svalues.female.cross_vs_idac"].detail
    assert "5 pairs" in st["svalues.female.cross_vs_opendose"].detail
    # the figures behind the claim, recomputed independently of the check
    ds, reg = load_svalues("lu177_adult_female"), get_registry()
    conv = 1000.0 / 3600.0
    for target, key in [("Spleen", "spleen"), ("Pancreas", "pancreas"), ("Brain", "brain")]:
        mine = ds.s(target, "Liver") * conv
        for who in ("idac", "opendose"):
            pub = reg.value(f"opendose2020.t1.lu177.{key}.{who}")
            assert mine == pytest.approx(pub, rel=0.006), (target, who)


def test_opendose_liver_self_term_is_a_declared_known_conflict_and_not_used():
    st = _status(get_registry())["svalues.female.liver_self_vs_opendose"]
    assert st.status == KNOWN_CONFLICT and "22" in st.detail  # MIRDcalc is 22.5 % below OpenDose
    assert "opendose2020.t1.lu177.liver.opendose" in get_registry().excluded
    assert "ICRP 133" in get_registry().excluded["opendose2020.t1.lu177.liver.opendose"]
    # while IDAC-Dose and MIRDcalc agree on the same quantity
    assert get_registry().value("opendose2020.t1.lu177.liver.idac") == pytest.approx(
        load_svalues("lu177_adult_female").self_s("Liver") * 1000 / 3600, rel=0.002
    )


def test_known_conflict_with_an_external_claim_does_not_distrust_our_dataset():
    dc.require_trusted_dataset("lu177_adult_female")  # has a KNOWN_CONFLICT check: still trusted
    dc.require_trusted_dataset("lu177_adult_male")


def test_a_known_conflict_that_disappears_does_block_the_dataset(monkeypatch):
    raw = yaml.safe_load(_claims_text())
    for c in raw["checks"]:
        if c["id"] == "svalues.female.liver_self_vs_opendose":
            c["rel_tol"] = 0.5  # now the 22 % difference passes, so the declared conflict is stale
    reg = load_registry(yaml.safe_dump(raw))
    assert _status(reg)["svalues.female.liver_self_vs_opendose"].status == UNEXPECTED_PASS
    monkeypatch.setattr(dc, "get_registry", lambda: reg)
    with pytest.raises(ValueError, match="UNEXPECTED_PASS"):
        dc.require_trusted_dataset("lu177_adult_female")


# --- mutation tests: each new check can fail --------------------------------------------------------


def test_a_wrong_cross_term_trips_the_idac_and_opendose_checks(monkeypatch):
    reg = _with_dataset(
        monkeypatch,
        "lu177_adult_female",
        lambda d: d["s_total"]["Spleen"].update(Liver=1.03 * d["s_total"]["Spleen"]["Liver"]),
    )
    st = _status(reg)
    assert st["svalues.female.cross_vs_idac"].status == FAIL
    assert st["svalues.female.cross_vs_opendose"].status == FAIL
    assert "Spleen <- Liver" in st["svalues.female.cross_vs_idac"].detail


def test_a_wrong_cross_share_trips_the_eanm_check(monkeypatch):
    def tenfold(d):
        for t in ("Kidneys",):
            for s in ("Liver", "Spleen"):
                d["s_total"][t][s] *= 10
                d["s_total"][s][t] *= 10  # keep reciprocity so only this check can catch it

    reg = _with_dataset(monkeypatch, "lu177_adult_male", tenfold)
    assert _status(reg)["svalues.male.kidney_cross_share_vs_eanm"].status == FAIL


def test_the_scenario_organ_reciprocity_check_is_strict(monkeypatch):
    reg = _with_dataset(
        monkeypatch,
        "lu177_adult_male",
        lambda d: d["s_total"]["Liver"].update(Spleen=1.03 * d["s_total"]["Liver"]["Spleen"]),
    )
    st = _status(reg)
    assert st["svalues.reciprocity_scenario_organs"].status == FAIL  # 3 % > 1 %
    assert st["svalues.reciprocity"].status == PASS  # the general 5 % check would not notice


def test_reciprocity_ignores_noise_dominated_terms_but_not_real_asymmetry(monkeypatch):
    # a huge relative asymmetry in a tiny cross term (Brain <-> Pancreas, ~1e-5 of the self terms)
    reg = _with_dataset(
        monkeypatch,
        "lu177_adult_female",
        lambda d: d["s_total"]["Brain"].update(Pancreas=5 * d["s_total"]["Brain"]["Pancreas"]),
    )
    st = _status(reg)["svalues.female.reciprocity"]
    assert st.status == PASS and "below the noise floor" in st.detail
    # the same factor on a cross term well above the floor is caught
    reg = _with_dataset(
        monkeypatch,
        "lu177_adult_female",
        lambda d: d["s_total"]["Kidneys"].update(Liver=2 * d["s_total"]["Kidneys"]["Liver"]),
    )
    assert _status(reg)["svalues.female.reciprocity"].status == FAIL


def test_reciprocity_with_no_testable_pair_fails_closed():
    raw = yaml.safe_load(_claims_text())
    for c in raw["checks"]:
        if c["id"] == "svalues.female.reciprocity":
            c["min_relative"] = 10.0  # nothing is this large
    st = _status(load_registry(yaml.safe_dump(raw)))["svalues.female.reciprocity"]
    assert st.status == FAIL and "no pair above the noise floor" in st.detail


# --- primary-source nuclear data -------------------------------------------------------------------


def test_half_life_is_the_current_ddep_value_and_the_old_one_is_within_its_uncertainty():
    reg = get_registry()
    assert reg.value("ddep.lu177.half_life") == pytest.approx(6.6443)
    assert reg.value("ddep.lu177.half_life_unc") == pytest.approx(0.0009)
    assert reg.value("staanum2021.lu177.half_life") == pytest.approx(6.647)
    st = _status(reg)
    assert st["nuclide.half_life_staanum_vs_ddep"].status == PASS
    assert st["nuclide.half_life_within_older_uncertainty"].status == PASS


def test_electron_energy_per_decay_lies_inside_the_bounds_built_from_the_ddep_emission_tables():
    st = _status(get_registry())
    for key in ("nuclide.delta_e_within_ddep_bounds", "nuclide.delta_e_nndc_within_ddep_bounds"):
        assert st[key].status == PASS, st[key].detail
    # the bounds themselves (keV): beta averages 132.7 + conversion/Auger between 8.0 and 18.3
    reg = get_registry()
    beta = (
        sum(
            reg.value(f"ddep.lu177.beta{k}.prob") * reg.value(f"ddep.lu177.beta{k}.avg")
            for k in ("03", "02", "01", "00")
        )
        / 100
    )
    assert beta == pytest.approx(132.7, abs=0.1)
    assert "140.7" in st["nuclide.delta_e_within_ddep_bounds"].detail
    assert "150.99" in st["nuclide.delta_e_within_ddep_bounds"].detail


def test_an_implausible_electron_energy_is_rejected():
    raw = yaml.safe_load(_claims_text())
    raw["claims"]["gleisner2022.delta_e"]["value"] = 0.165  # 165 keV: above the DDEP upper bound
    st = _status(load_registry(yaml.safe_dump(raw)))
    assert st["nuclide.delta_e_within_ddep_bounds"].status == FAIL


def test_a_stale_registry_value_is_caught(monkeypatch):
    stale = dataclasses.replace(nuclides_mod.NUCLIDES["Lu-177"], half_life_d=6.647)
    monkeypatch.setitem(nuclides_mod.NUCLIDES, "Lu-177", stale)
    st = _status(load_registry())["nuclides.registry_matches_claims"]
    assert st.status == FAIL and "half_life_d=6.647" in st.detail


def test_the_shipped_registry_matches_the_claims():
    assert _status(get_registry())["nuclides.registry_matches_claims"].status == PASS


# --- bookkeeping ----------------------------------------------------------------------------------------


def test_every_dataset_in_the_package_has_checks_and_is_trusted():
    for name in refdata.available_svalue_sets():
        assert any(r.dataset == name for r in get_registry().results), name
        dc.require_trusted_dataset(name)


def test_summary_counts_of_the_shipped_registry():
    reg = get_registry()
    counts = {s: sum(r.status == s for r in reg.results) for s in (PASS, KNOWN_CONFLICT, FAIL)}
    assert counts[FAIL] == 0 and counts[KNOWN_CONFLICT] == 6
    assert len(reg.claims) >= 130 and len(reg.results) >= 40
    assert reg.ok
