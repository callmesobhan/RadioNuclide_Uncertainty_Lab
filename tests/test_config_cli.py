import copy
import json
from pathlib import Path

import pytest
import yaml

from RadioNuclide_Uncertainty_Lab import load_config
from RadioNuclide_Uncertainty_Lab.cli import main

EXAMPLE = Path(__file__).parents[1] / "examples" / "configs" / "lu177_dotatate_multi_organ.yaml"
FAST = ["--n", "1024", "--sobol-n", "256"]


def _raw():
    return yaml.safe_load(EXAMPLE.read_text())


def test_example_config_loads_with_expected_structure():
    cfg = load_config(EXAMPLE)
    assert cfg.model.outputs == (
        "kidneys",
        "liver",
        "spleen",
        "kidneys_to_liver",
        "kidneys_to_spleen",
    )
    groups = {f.name for f in cfg.model.inputs.factors}
    assert {"dose_calibrator", "spect_quantification", "tac_fit.kidneys", "tac_fit.liver"} <= groups
    assert len(groups) == 14
    assert cfg.run.seed == 20261012
    assert len(cfg.sha256) == 64


def test_organ_templates_expand_per_organ_and_respect_the_organs_filter():
    cfg = load_config(EXAMPLE)
    names = set(cfg.model.inputs.names)
    for organ in ("kidneys", "liver", "spleen"):
        assert f"{organ}.activity_scale.pvc" in names
    assert "liver.activity_scale.tac_fit" in names and "spleen.activity_scale.tac_fit" in names
    assert "kidneys.activity_scale.tac_fit" not in names  # restricted by `organs:`
    # the kidney fit pair is correlated and lives in one factor
    factor = next(f for f in cfg.model.inputs.factors if f.name == "tac_fit.kidneys")
    assert factor.members == ("kidneys.uptake_fraction", "kidneys.effective_half_life_h")


def test_config_hash_changes_with_content():
    raw = _raw()
    other = copy.deepcopy(raw)
    other["run"]["seed"] = 1  # not tied to a published claim
    assert load_config(raw).sha256 != load_config(other).sha256
    assert load_config(raw).sha256 == load_config(copy.deepcopy(raw)).sha256


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(typo=1), "unknown top-level keys"),
        (lambda r: r.update(model="nope"), "unknown model"),
        (lambda r: r["run"].update(nsamples=5), "unknown run settings"),
        (lambda r: r["run"].update(method="halton"), "run.method"),
        (
            lambda r: r["parameters"]["activity_scale.spect_stability"].update(sdd=1),
            "spect_stability",
        ),
        (lambda r: r.pop("parameters"), "missing required key"),
        (lambda r: r.pop("s_values"), "needs the top-level 's_values'"),
        (lambda r: r["organs"]["liver"].update(region="Livr"), "not in dataset"),
        (lambda r: r["organs"]["liver"].update(mass_ref_kg=2.0), "either 'region' or explicit"),
        (
            lambda r: r["parameters"]["activity_scale.spect_reproduction"].update(rel_sd=0.07),
            "does not match claim",
        ),
        (
            lambda r: r["parameters"]["activity_scale.spect_reproduction"].update(
                claim="staanum2021.one_spect.lambda_unc.text"
            ),
            "excluded",
        ),
        (
            lambda r: r["parameters"]["activity_scale.spect_reproduction"].update(
                claim="no.such.claim"
            ),
            "unknown claim",
        ),
        (lambda r: r["correlations"][0].update(rho=-0.5), "does not match claim"),
        (lambda r: r.pop("organs"), "no organs|unknown quantities|at least one organ"),
        (
            lambda r: r["parameters"]["{organ}.activity_scale.pvc"].update(organs=["kidney"]),
            "unknown organs",
        ),
        (
            lambda r: r["parameters"].update(
                {"activity_scale.x": {"organs": ["liver"], "dist": "normal", "mean": 1, "sd": 0.1}}
            ),
            "only valid with",
        ),
        (  # typo in the scale-factor kind: the label after it is free-form, the kind is not
            lambda r: r["parameters"].update(
                {"kidneys.activity_scal.pvc2": {"dist": "normal", "mean": 1, "sd": 0.1}}
            ),
            "unknown quantities",
        ),
        (  # typo in the organ name
            lambda r: r["parameters"].update(
                {"kidney.activity_scale.pvc2": {"dist": "normal", "mean": 1, "sd": 0.1}}
            ),
            "unknown quantities",
        ),
        (
            lambda r: r.update(
                correlations=[
                    {
                        "between": ["kidneys.activity_scale.pvc", "liver.activity_scale.pvc"],
                        "rho": 0.3,
                    }
                ]
            ),
            "share a group",
        ),
    ],
)
def test_bad_configs_fail_loudly(mutate, message):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        load_config(raw)


def test_cli_run_writes_report_files(tmp_path, capsys):
    assert main(["run", str(EXAMPLE), "--out", str(tmp_path), *FAST, "--no-plots"]) == 0
    data = json.loads((tmp_path / "result.json").read_text())
    assert data["provenance"]["seed"] == 20261012
    assert data["provenance"]["config_sha256"] == load_config(EXAMPLE).sha256
    assert data["provenance"]["package"] == "RadioNuclide_Uncertainty_Lab"
    assert {"outputs", "rank_sensitivity", "error_budget", "inputs", "budget_matrix_st"} <= set(
        data
    )
    assert set(data["outputs"]) == set(data["error_budget"]) == set(data["rank_sensitivity"])
    # tiny design => don't assert the order of comparable factors; the dose-calibrator term
    # (1 % + 1 %) must come out near the bottom for every organ dose
    kid = data["error_budget"]["kidneys"]["factors"]
    assert {f["factor"] for f in kid[-3:]} <= {
        "dose_calibrator",
        "tac_model.kidneys",
        "dose_factor.kidneys",
    } | {f["factor"] for f in kid if f["factor"].endswith((".liver", ".spleen"))}
    report = (tmp_path / "report.md").read_text()
    for needle in (
        "Error budget",
        "factors × outputs",
        "Provenance",
        "Correlation between outputs",
    ):
        assert needle in report
    out = capsys.readouterr().out
    assert "kidneys_to_liver" in out and "largest sources for kidneys" in out


def test_cli_run_is_deterministic(tmp_path):
    main(["run", str(EXAMPLE), "--out", str(tmp_path / "a"), *FAST, "--no-plots"])
    main(["run", str(EXAMPLE), "--out", str(tmp_path / "b"), *FAST, "--no-plots"])
    a = json.loads((tmp_path / "a" / "result.json").read_text())
    b = json.loads((tmp_path / "b" / "result.json").read_text())
    for d in (a, b):
        d["provenance"].pop("created_utc")
    assert a == b


def test_cli_plots_are_created_for_multi_output_runs(tmp_path):
    pytest.importorskip("matplotlib")
    assert main(["run", str(EXAMPLE), "--out", str(tmp_path), *FAST]) == 0
    for name in (
        "intervals.png",
        "budget_matrix.png",
        "outputs/kidneys/budget.png",
        "outputs/kidneys_to_liver/distribution.png",
        "outputs/spleen/rank.png",
    ):
        assert (tmp_path / name).stat().st_size > 5_000, name


def test_cli_validate_models_and_errors(tmp_path, capsys):
    assert main(["validate", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "outputs: kidneys, liver, spleen" in out and "kidneys.uptake_fraction" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: multi_organ\nparameters: {}\nfoo: 1\n")
    assert main(["validate", str(bad)]) == 2
    assert "unknown top-level keys" in capsys.readouterr().err
    assert main(["models"]) == 0
    assert capsys.readouterr().out.strip() == "multi_organ"
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2


def test_cli_check_data_reports_and_exits_zero(capsys):
    assert main(["check-data"]) == 0
    out = capsys.readouterr().out
    assert "KNOWN_CONFLICT" in out and "Excluded claims" in out and "svalues.reciprocity" in out


def test_cli_check_data_exits_nonzero_when_a_check_fails(monkeypatch, capsys):
    from RadioNuclide_Uncertainty_Lab import datacheck as dc

    broken = dc.load_registry(
        "sources: {s: {citation: x, id: y}}\n"
        "claims: {a: {value: 1, unit: u, source: s, locator: l}, b: {value: 2, unit: u, source: s, locator: l}}\n"
        "checks: [{id: c, kind: agree, a: a, b: b, rel_tol: 0.01, expect: pass}]\n"
    )
    monkeypatch.setattr(dc, "get_registry", lambda: broken)
    assert main(["check-data"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_report_and_json_carry_diagnostics_and_data_integrity(tmp_path):
    assert main(["run", str(EXAMPLE), "--out", str(tmp_path), *FAST, "--no-plots"]) == 0
    data = json.loads((tmp_path / "result.json").read_text())
    di = data["data_integrity"]
    assert len(di["claims_used"]) >= 12 and "staanum2021.t2.pvc" in di["claims_used"]
    assert di["checks_known_conflict"] == 6 and len(di["excluded_claims"]) == 9
    assert di["s_value_dataset"]["dataset"] == "lu177_adult_male"
    assert di["s_value_dataset"]["cross_dose"] is True
    assert isinstance(data["diagnostics"], list)
    report = (tmp_path / "report.md").read_text()
    assert "## Diagnostics" in report and "## Data integrity" in report
    assert "cross-organ dose included" in report


def test_cli_warns_on_stderr_about_unreproducible_runs(tmp_path, capsys):
    raw = _raw()
    raw["run"].pop("seed")
    cfg = tmp_path / "noseed.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    assert main(["run", str(cfg), "--out", str(tmp_path / "o"), *FAST, "--no-plots"]) == 0
    err = capsys.readouterr().err
    assert "warning [NO_SEED]" in err
