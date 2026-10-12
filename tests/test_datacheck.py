"""The claims registry and the consistency checks that exclude conflicting data."""

import textwrap

import pytest
import yaml

from RadioNuclide_Uncertainty_Lab import ExcludedClaimError, get_registry, normal
from RadioNuclide_Uncertainty_Lab import datacheck as dc
from RadioNuclide_Uncertainty_Lab.datacheck import (
    FAIL,
    KNOWN_CONFLICT,
    PASS,
    UNEXPECTED_PASS,
    evaluate_expression,
    load_registry,
    verify_correlation_claim,
    verify_parameter_claim,
)

EXPECTED_KNOWN_CONFLICTS = {
    "staanum.abstract_vs_results.dotatoc_mean",
    "staanum.abstract_vs_results.dotatoc_median",
    "staanum.one_spect_lambda_uncertainty",
    "gear.s_model_c2_uncertainty",
    "gear.s_model_c1_uncertainty",
    "svalues.female.liver_self_vs_opendose",
}


def _registry(claims: str, checks: str):
    header = "sources: {s: {citation: x, id: y}}\n"
    return load_registry(
        header
        + "claims:\n"
        + textwrap.indent(textwrap.dedent(claims), "  ")
        + "checks:\n"
        + textwrap.indent(textwrap.dedent(checks), "  ")
    )


# --- the shipped registry ---------------------------------------------------------------------


def test_shipped_registry_passes_and_knows_exactly_the_expected_discrepancies():
    reg = get_registry()
    by_status = {}
    for r in reg.results:
        by_status.setdefault(r.status, set()).add(r.id)
    assert reg.ok
    assert by_status.get(FAIL, set()) == set() and by_status.get(UNEXPECTED_PASS, set()) == set()
    assert by_status[KNOWN_CONFLICT] == EXPECTED_KNOWN_CONFLICTS
    assert len(by_status[PASS]) >= 30


def test_conflicting_claims_are_excluded_and_unusable():
    reg = get_registry()
    assert set(reg.excluded) == {
        "staanum2021.dotatoc.specific_dose.mean.abstract",
        "staanum2021.dotatoc.specific_dose.mean.results",
        "staanum2021.dotatoc.specific_dose.median.abstract",
        "staanum2021.dotatoc.specific_dose.median.results",
        "staanum2021.one_spect.lambda_unc.text",
        "staanum2021.one_spect.lambda_unc.table2",
        "gear2018.s_model.c1_unc_listed",
        "gear2018.s_model.c2_unc_listed",
        "opendose2020.t1.lu177.liver.opendose",
    }
    with pytest.raises(ExcludedClaimError, match="swap"):
        reg.require("staanum2021.dotatoc.specific_dose.mean.abstract")
    # the numbers the *consistent* part of Gear Table 10 provides stay usable
    assert reg.value("gear2018.s_model.c2") == pytest.approx(0.961)


def test_every_claim_has_a_known_source_and_a_location():
    reg = get_registry()
    for c in reg.claims.values():
        assert c.source in reg.sources and c.locator and c.unit


def test_unknown_claim_is_a_key_error():
    with pytest.raises(KeyError, match="unknown claim"):
        get_registry().get("staanum2021.nope")


def test_report_lists_status_and_exclusions():
    text = get_registry().format_report()
    assert "KNOWN_CONFLICT" in text and "Excluded claims" in text and "claims," in text


# --- safe expression evaluator ------------------------------------------------------------------


def test_expression_evaluator_arithmetic_and_claim_names():
    vals = {"a.b": 3.0, "c.d": 4.0}
    assert evaluate_expression("sqrt(a.b**2 + c.d**2)", vals.__getitem__) == pytest.approx(5.0)
    assert evaluate_expression("-(a.b) / 2 + 1e-1", vals.__getitem__) == pytest.approx(-1.4)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "a.b.__class__",
        "[1, 2][0]",
        "a.b if True else 0",
        "lambda: 1",
        "sqrt(1, 2)",
        "open('x')",
        "2 ** 100",
        "abs(a.b) < 2",
    ],
)
def test_expression_evaluator_rejects_anything_but_arithmetic(expr):
    known = {"a.b": 1.0}  # an unknown dotted name is a claim lookup, which fails (KeyError)
    with pytest.raises((ValueError, SyntaxError, TypeError, KeyError)):
        evaluate_expression(expr, known.__getitem__)


# --- check kinds and the exclusion mechanism ----------------------------------------------------


CLAIMS = """
x.a: {value: 10.0, unit: u, source: s, locator: here}
x.b: {value: 10.5, unit: u, source: s, locator: here}
x.c: {value: 20.0, unit: u, source: s, locator: here}
x.t1: {value: 3.0, unit: u, source: s, locator: here}
x.t2: {value: 4.0, unit: u, source: s, locator: here}
x.tot: {value: 5.0, unit: u, source: s, locator: here}
"""


def test_agree_pass_fail_and_default_exclusion():
    reg = _registry(
        CLAIMS,
        """
        - {id: ok, kind: agree, a: x.a, b: x.b, rel_tol: 0.1, expect: pass}
        - {id: bad, kind: agree, a: x.a, b: x.c, rel_tol: 0.1, expect: pass}
    """,
    )
    status = {r.id: r.status for r in reg.results}
    assert status == {"ok": PASS, "bad": FAIL}
    assert not reg.ok
    assert set(reg.excluded) == {"x.a", "x.c"}  # operands of the failed check
    with pytest.raises(ExcludedClaimError):
        reg.require("x.c")
    assert reg.require("x.b").value == 10.5  # unaffected


def test_expect_conflict_requires_the_conflict_to_persist():
    reg = _registry(
        CLAIMS,
        """
        - {id: known, kind: agree, a: x.a, b: x.c, rel_tol: 0.1, expect: conflict}
        - {id: fixed_upstream, kind: agree, a: x.a, b: x.b, rel_tol: 0.1, expect: conflict}
    """,
    )
    status = {r.id: r.status for r in reg.results}
    assert status == {"known": KNOWN_CONFLICT, "fixed_upstream": UNEXPECTED_PASS}
    assert not reg.ok  # the resolved discrepancy must be reviewed, not silently ignored
    assert set(reg.excluded) == {"x.a", "x.c"}  # only the still-conflicting check excludes


def test_explicit_exclude_list_overrides_the_default():
    reg = _registry(
        CLAIMS,
        """
        - id: only_c
          kind: agree
          a: x.c
          b: {expr: "x.a + x.b"}
          rel_tol: 0.01
          expect: conflict
          exclude: [x.c]
    """,
    )
    assert set(reg.excluded) == {"x.c"}


def test_quadrature_modes():
    reg = _registry(
        CLAIMS,
        """
        - {id: exact, kind: quadrature, terms: [x.t1, x.t2], total: x.tot, mode: nearest, tol: 0.01, expect: pass}
        - {id: below, kind: quadrature, terms: [x.t1, x.t2], total: x.t1, mode: up, tol: 1.0, expect: pass}
        - {id: rounded_up, kind: quadrature, terms: [x.t1, x.t2], total: x.c, mode: up, tol: 20, expect: pass}
    """,
    )
    assert {r.id: r.status for r in reg.results} == {
        "exact": PASS,
        "below": FAIL,
        "rounded_up": PASS,
    }


def test_bound_and_spread_kinds():
    reg = _registry(
        CLAIMS,
        """
        - {id: in, kind: bound, value: x.a, lo: 0, hi: 11, expect: pass}
        - {id: out, kind: bound, value: x.c, lo: 0, hi: 11, expect: pass}
        - {id: tight, kind: spread, values: [x.a, x.b], max_ratio: 1.1, expect: pass}
        - {id: wide, kind: spread, values: [x.a, x.c], max_ratio: 1.1, expect: pass}
    """,
    )
    assert {r.id: r.status for r in reg.results} == {
        "in": PASS,
        "out": FAIL,
        "tight": PASS,
        "wide": FAIL,
    }


def test_unevaluable_check_fails_closed():
    reg = _registry(
        CLAIMS,
        """
        - {id: dangling, kind: agree, a: x.a, b: x.missing, rel_tol: 0.1, expect: pass}
    """,
    )
    (r,) = reg.results
    assert r.status == FAIL and "could not evaluate" in r.detail
    assert "x.a" in reg.excluded


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "sources: {s: {citation: x, id: y}}\nclaims: {a: {value: 1, unit: u, source: nope, locator: l}}\nchecks: []\n",
            "unknown source",
        ),
        (
            "sources: {s: {citation: x, id: y}}\nclaims: {a: {value: 1, unit: u, source: s, locator: l, extra: 1}}\nchecks: []\n",
            "unknown keys",
        ),
        (
            "sources: {s: {citation: x, id: y}}\nclaims: {}\nchecks: [{id: c, kind: bound, value: 1}, {id: c, kind: bound, value: 1}]\n",
            "duplicate check ids",
        ),
    ],
)
def test_malformed_registries_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        load_registry(text)


def test_invalid_expect_value_is_rejected():
    reg = _registry(CLAIMS, "- {id: z, kind: bound, value: x.a, lo: 0, hi: 1, expect: maybe}\n")
    with pytest.raises(ValueError, match="expect must be"):
        _ = reg.results


# --- reference-dataset checks fail when the data is wrong ------------------------------------------


def _patched_dataset(monkeypatch, mutate):
    import copy
    import json
    from importlib import resources

    from RadioNuclide_Uncertainty_Lab import refdata

    raw = json.loads(
        (
            resources.files("RadioNuclide_Uncertainty_Lab") / "data" / "lu177_adult_male.json"
        ).read_text()
    )
    broken = copy.deepcopy(raw)
    mutate(broken)
    monkeypatch.setattr(
        refdata, "cached_svalues", lambda name: refdata._validate_and_build(broken, name)
    )
    return load_registry()


def test_reciprocity_check_catches_an_asymmetric_matrix(monkeypatch):
    reg = _patched_dataset(
        monkeypatch,
        lambda d: d["s_total"]["Kidneys"].update(Liver=2 * d["s_total"]["Kidneys"]["Liver"]),
    )
    status = {r.id: r.status for r in reg.results}
    assert status["svalues.reciprocity"] == FAIL


def test_energy_conservation_check_catches_impossible_values(monkeypatch):
    reg = _patched_dataset(
        monkeypatch,
        lambda d: d["s_total"]["Kidneys"].update(Kidneys=10 * d["s_total"]["Kidneys"]["Kidneys"]),
    )
    assert {r.id: r.status for r in reg.results}["svalues.energy_conservation"] == FAIL


def test_cross_database_check_catches_a_wrong_self_value(monkeypatch):
    reg = _patched_dataset(
        monkeypatch,
        lambda d: d["s_total"]["Spleen"].update(Spleen=1.1 * d["s_total"]["Spleen"]["Spleen"]),
    )
    assert {r.id: r.status for r in reg.results}["svalues.agree_with_idac_self_s"] == FAIL


def test_untrusted_dataset_is_refused(monkeypatch):
    reg = _patched_dataset(
        monkeypatch,
        lambda d: d["s_total"]["Kidneys"].update(Liver=3 * d["s_total"]["Kidneys"]["Liver"]),
    )
    monkeypatch.setattr(dc, "get_registry", lambda: reg)
    with pytest.raises(ValueError, match="failed consistency checks.*reciprocity"):
        dc.require_trusted_dataset("lu177_adult_male")
    with pytest.raises(ValueError, match="no registered consistency checks"):
        dc.require_trusted_dataset("some_unchecked_dataset")


# --- claims tied to config entries ------------------------------------------------------------------


def test_parameter_must_realise_its_claim():
    verify_parameter_claim("p", normal(1, rel_sd=0.07), "staanum2021.t2.pvc")  # 7 %
    with pytest.raises(ValueError, match="does not match claim"):
        verify_parameter_claim("p", normal(1, rel_sd=0.08), "staanum2021.t2.pvc")
    with pytest.raises(ExcludedClaimError):
        verify_parameter_claim(
            "p", normal(1, rel_sd=0.055), "staanum2021.one_spect.lambda_unc.text"
        )
    with pytest.raises(ValueError, match="zero-mean"):
        verify_parameter_claim("p", normal(0, 1), "staanum2021.t2.pvc")


def test_correlation_must_realise_its_claim():
    verify_correlation_claim(("a", "b"), -0.76, "staanum2021.derived.rho_c0_teff")
    with pytest.raises(ValueError, match="does not match claim"):
        verify_correlation_claim(("a", "b"), -0.5, "staanum2021.derived.rho_c0_teff")


def test_claims_file_is_valid_yaml_with_expected_top_level_keys():
    from importlib import resources

    raw = yaml.safe_load(
        (resources.files("RadioNuclide_Uncertainty_Lab") / "data" / "claims.yaml").read_text()
    )
    assert set(raw) == {"version", "sources", "claims", "checks"}
