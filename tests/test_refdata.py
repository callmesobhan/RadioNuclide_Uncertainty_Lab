"""The packaged S-value dataset: structure, provenance and physics."""

import copy
import json
from importlib import resources

import pytest

from RadioNuclide_Uncertainty_Lab import load_svalues
from RadioNuclide_Uncertainty_Lab.refdata import (
    _validate_and_build,
    available_svalue_sets,
    cached_svalues,
)

NAME = "lu177_adult_male"


@pytest.fixture(scope="module")
def raw():
    return json.loads(
        (resources.files("RadioNuclide_Uncertainty_Lab") / "data" / f"{NAME}.json").read_text()
    )


def test_dataset_is_available_and_loads():
    assert NAME in available_svalue_sets()
    ds = load_svalues(NAME)
    assert ds.nuclide == "Lu-177" and ds.phantom == "ICRP Adult Male"
    assert set(ds.regions) >= {"Kidneys", "Liver", "Spleen"}
    assert cached_svalues(NAME) is cached_svalues(NAME)


def test_provenance_pins_the_exact_source_file(raw):
    prov = raw["provenance"]
    assert prov["member_crc32"] == "4302f61d"
    assert len(prov["member_sha256"]) == 64
    assert prov["archive_member"].endswith("SDATA_Lu-177_ICRP Adult Male.csv")
    assert "MIRD Pamphlet No. 28" in prov["citation"]
    assert "noncommercial" in prov["terms"]
    assert prov["bytes_fetched"] < 5_000_000  # a few MB of a 548 MB archive: nothing else was kept
    assert load_svalues(NAME).fingerprint == prov["member_sha256"][:12]


def test_orientation_is_target_row_source_column(raw):
    ds = load_svalues(NAME)
    # a source can only deposit energy elsewhere through photons: cross terms are ~1000x smaller
    assert ds.s("Liver", "Kidneys") < 1e-2 * ds.s("Kidneys", "Kidneys")
    assert ds.cross_table({"k": "Kidneys", "l": "Liver"}) == {
        "k": {"l": ds.s("Kidneys", "Liver")},
        "l": {"k": ds.s("Liver", "Kidneys")},
    }


def test_masses_are_the_target_tissue_and_blood_masses():
    ds = load_svalues(NAME)
    assert ds.mass("Kidneys") == pytest.approx(0.422)
    assert ds.mass("Liver") == pytest.approx(2.360)
    assert ds.mass("Spleen") == pytest.approx(0.2284)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("s_total"), "missing keys"),
        (lambda d: d.update(dataset="other"), "declares itself"),
        (lambda d: d["units"].update(s_value="mGy/(MBq s)"), "units"),
        (lambda d: d["provenance"].pop("member_sha256"), "provenance lacks"),
        (lambda d: d["regions"].append("Liver"), "unique"),
        (lambda d: d["mass_kg"].update(Liver=0.0), "mass of 'Liver'"),
        (lambda d: d["mass_kg"].update(Liver=float("nan")), "mass of 'Liver'"),
        (lambda d: d["s_total"]["Liver"].update(Spleen=-1e-9), "non-negative"),
        (lambda d: d["s_total"]["Liver"].update(Spleen=float("inf")), "non-negative"),
        (lambda d: d["s_total"].pop("Spleen"), "targets differ"),
        (lambda d: d["s_total"]["Liver"].pop("Spleen"), "sources differ"),
        (lambda d: d["s_photon"]["Liver"].update(Liver=1.0), "photon S exceeds total"),
    ],
)
def test_corrupted_datasets_are_rejected(raw, mutate, message):
    broken = copy.deepcopy(raw)
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        _validate_and_build(broken, NAME)


def test_unknown_dataset_names_the_alternatives():
    with pytest.raises(KeyError, match="available"):
        load_svalues("nope")
