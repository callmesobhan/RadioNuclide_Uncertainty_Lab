"""Reproductions of published results.

Sources (open access; full citations in docs/data-sources.md):

* [Staanum2021] Staanum et al., EJNMMI Phys 8:78 - kidney uncertainty budget, Table 2.
* [Gear2018]    Gear et al., EJNMMI 45:2456 - EANM uncertainty guidance, worked example.
* [Gleisner2022] Sjogreen Gleisner et al., EJNMMI 49:1778 - S-values (see test_dosimetry.py).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from RadioNuclide_Uncertainty_Lab import (
    InputSpace,
    Parameter,
    UncertainModel,
    load_config,
    lognormal,
    propagate,
    sobol_indices,
    truncated_normal,
)
from RadioNuclide_Uncertainty_Lab.datacheck import get_registry
from RadioNuclide_Uncertainty_Lab.kinetics import MonoExpFit

CONFIGS = Path(__file__).parents[1] / "examples" / "configs"
SCENARIO = CONFIGS / "lu177_dotatate_multi_organ.yaml"

# --------------------------------------------------------------------------------------------
# [Staanum2021] Table 2, 3-SPECT kidney example. Standard uncertainties in percent. The 3 %
# calibration-source term is omitted: the paper notes it cancels for specific absorbed dose.
# --------------------------------------------------------------------------------------------
REGISTRY = get_registry()


def _claim(claim_id: str) -> float:
    return REGISTRY.require(claim_id).value  # refuses excluded claims


_T2 = "staanum2021.t2."
STAANUM_SHARED = {  # one error common to every organ measured on the same system
    "dose calibrator calibration": _claim(_T2 + "dose_calibrator_calibration"),
    "dose calibrator stability": _claim(_T2 + "dose_calibrator_stability"),
    "scanner calibration (S)": _claim(_T2 + "scanner_sensitivity"),
    "scanner calibration (dead-time constant)": _claim(_T2 + "scanner_deadtime_constant"),
    "reproduction of activity": _claim(_T2 + "reproduction"),
    "scanner stability": _claim(_T2 + "scanner_stability"),
    "dead-time correction": _claim(_T2 + "deadtime_correction"),
}
STAANUM_ORGAN_SPECIFIC = {
    "delineation": _claim(_T2 + "delineation"),
    "partial-volume correction": _claim(_T2 + "pvc"),
    "single-exponential model (early)": _claim(_T2 + "exp_model_early"),
    "single-exponential model (late)": _claim(_T2 + "exp_model_late"),
    "time-activity fit (AUC, Fig. 6b example)": _claim("staanum2021.fig6b.auc"),
    "dose factor (gamma absorption)": _claim(_T2 + "dose_factor_gamma"),
    "dose factor (density)": _claim(_T2 + "dose_factor_density"),
}


def _quadrature(components) -> float:
    return math.sqrt(sum(c**2 for c in components.values())) / 100


@pytest.fixture(scope="module")
def scenario():
    return load_config(SCENARIO)


@pytest.fixture(scope="module")
def propagated(scenario):
    return propagate(scenario.model, n=2**16, seed=7)


def test_staanum_quadrature_total_is_12_1_percent_which_the_paper_rounds_up_to_13():
    total = _quadrature({**STAANUM_SHARED, **STAANUM_ORGAN_SPECIFIC})
    assert total == pytest.approx(0.1213, abs=0.0005)
    assert total < 0.13 <= math.ceil(total * 100) / 100  # paper: "13 % (rounded up)"


def test_monte_carlo_reproduces_the_published_kidney_uncertainty(propagated):
    cv = propagated.summary("kidneys").cv
    expected = _quadrature({**STAANUM_SHARED, **STAANUM_ORGAN_SPECIFIC})
    assert cv == pytest.approx(expected, abs=0.004)
    assert 0.115 < cv <= 0.13


def test_same_budget_is_applied_to_every_organ(propagated):
    cvs = [propagated.summary(o).cv for o in ("kidneys", "liver", "spleen")]
    assert max(cvs) - min(cvs) < 0.004


def test_shared_errors_cancel_in_ratios_and_organ_specific_ones_add(propagated):
    # ratio of two organs measured on the same system: only organ-specific terms remain,
    # once for each organ
    expected = math.sqrt(2) * _quadrature(STAANUM_ORGAN_SPECIFIC)
    for ratio in ("kidneys_to_liver", "kidneys_to_spleen"):
        assert propagated.summary(ratio).cv == pytest.approx(expected, abs=0.006)
    assert propagated.summary("kidneys_to_liver").cv > propagated.summary("kidneys").cv


def test_shared_errors_correlate_organ_doses_by_their_share_of_variance(propagated):
    shared = _quadrature(STAANUM_SHARED) ** 2
    total = _quadrature({**STAANUM_SHARED, **STAANUM_ORGAN_SPECIFIC}) ** 2
    assert propagated.correlation("kidneys", "liver") == pytest.approx(shared / total, abs=0.03)


def test_sobol_budget_matches_the_quadrature_shares(scenario):
    sob = sobol_indices(scenario.model, n=2**13, seed=3, output="kidneys")
    st = dict(zip(sob.factors, sob.st, strict=True))
    total = sum(c**2 for c in {**STAANUM_SHARED, **STAANUM_ORGAN_SPECIFIC}.values())
    expected = {
        "delineation_pvc.kidneys": (3**2 + 7**2) / total,
        "spect_quantification": (2**2 + 1 + 6**2 + 2**2 + 1) / total,
        "tac_fit.kidneys": 6.1**2 / total,
        "dose_calibrator": 2 / total,
        "tac_model.kidneys": 2 / total,
        "dose_factor.kidneys": 2 / total,
    }
    for factor, share in expected.items():
        assert st[factor] == pytest.approx(share, abs=0.03), factor
    # organ-specific factors of *other* organs cannot influence the kidney dose
    for factor in st:
        if factor.endswith((".liver", ".spleen")):
            assert st[factor] == pytest.approx(0.0, abs=0.02)


def test_nominal_organ_doses_are_anchored_to_the_published_cohort_means(scenario):
    medians = {r["name"]: np.array([r["median"]]) for r in scenario.model.inputs.describe()}
    dose = scenario.model.func(medians)
    gbq = 7.4  # administered activity of the scenario
    # kidneys: Staanum mean specific dose 0.46 Gy/GBq; liver, spleen: Santoro mean 0.54, 0.61
    assert dose["kidneys"][0] / gbq == pytest.approx(0.46, rel=0.005)
    assert dose["liver"][0] / gbq == pytest.approx(0.54, rel=0.005)
    assert dose["spleen"][0] / gbq == pytest.approx(0.61, rel=0.005)
    # independent corroboration: Staanum's kidney value agrees with Santoro's kidney cohort
    # (0.43 Gy/GBq) within 10 %, and the effective half-life implied by the inputs is 2.27 d
    assert dose["kidneys"][0] / gbq == pytest.approx(0.43, rel=0.10)
    assert medians["kidneys.effective_half_life_h"][0] / 24 == pytest.approx(2.27, abs=0.005)


# --------------------------------------------------------------------------------------------
# [Gear2018] worked example: 90Y-DOTATATE lesions. D = A~ x S, S = c1 m^-c2 (m in g),
# c1 = 0.429, c2 = 0.961. Inputs from Tables 1, 9, 10-13; r(A~, S) from Table 12.
# --------------------------------------------------------------------------------------------
C1, C2 = _claim("gear2018.s_model.c1"), _claim("gear2018.s_model.c2")


def _lesion(key: str):
    g = f"gear2018.{key}."
    return (
        _claim(g + "tia"),
        _claim(g + "tia_rel_unc") / 100,
        _claim(g + "mass"),
        _claim(g + "mass_rel_unc") / 100,
        _claim(g + "r_tia_s"),
        _claim(g + "dose"),
        _claim(g + "dose_rel_unc") / 100,
    )


LESIONS = {
    #          A~ (MBq h), u(A~)/A~, mass (g), u(m)/m, r(A~, S), published D (Gy), u(D)/D
    "liver lesion": _lesion("liver"),
    "pancreatic lesion": _lesion("pancreas"),
}


def _lognormal_param(name, nominal, rel, group):
    return Parameter(name, lognormal(nominal, math.exp(math.sqrt(math.log1p(rel**2)))), group=group)


def _lesion_model(lesion: str, scale: float = 1.0, marginal: str = "lognormal") -> UncertainModel:
    tia, u_tia, mass, u_mass, r, *_ = LESIONS[lesion]
    if marginal == "lognormal":
        params = [
            _lognormal_param("tia", tia, u_tia * scale, "chain"),
            _lognormal_param("mass", mass, u_mass * scale, "chain"),
        ]
    else:
        params = [
            Parameter("tia", truncated_normal(tia, rel_sd=u_tia * scale, low=0.0), group="chain"),
            Parameter(
                "mass", truncated_normal(mass, rel_sd=u_mass * scale, low=0.0), group="chain"
            ),
        ]
    # S falls with mass, so a *negative* r(A~, S) is a positive correlation of A~ with the mass
    return UncertainModel(
        lesion,
        InputSpace(params, {("tia", "mass"): -r}),
        lambda x: x["tia"] * C1 * x["mass"] ** (-C2),
        outputs="dose",
    )


@pytest.mark.parametrize("lesion", LESIONS)
def test_gear_nominal_dose_and_first_order_uncertainty_are_reproduced(lesion):
    tia, u_tia, mass, u_mass, r, d_pub, u_pub = LESIONS[lesion]
    assert tia * C1 * mass**-C2 == pytest.approx(d_pub, rel=0.005)
    u_s = C2 * u_mass  # S ~ m^-c2  =>  u(S)/S = c2 u(m)/m (their Table 10 lists 55.5 % / 25.5 %)
    first_order = math.sqrt(u_tia**2 + u_s**2 + 2 * r * u_tia * u_s)
    assert first_order == pytest.approx(u_pub, abs=0.003)


@pytest.mark.parametrize("lesion", LESIONS)
def test_monte_carlo_equals_first_order_propagation_in_the_linear_regime(lesion):
    tia, u_tia, mass, u_mass, r, *_ = LESIONS[lesion]
    scale = 0.05  # small relative uncertainties: first-order theory is then exact
    u_s = C2 * u_mass * scale
    first_order = math.sqrt((u_tia * scale) ** 2 + u_s**2 + 2 * r * u_tia * scale * u_s)
    cv = propagate(_lesion_model(lesion, scale), n=2**16, seed=1).summary().cv
    assert cv == pytest.approx(first_order, rel=0.02)


def test_monte_carlo_with_full_uncertainties_stays_near_the_published_value_when_lognormal():
    for lesion, tol in (("pancreatic lesion", 0.05), ("liver lesion", 0.10)):
        published = LESIONS[lesion][-1]
        cv = propagate(_lesion_model(lesion), n=2**17, seed=1).summary().cv
        assert cv == pytest.approx(published, rel=tol), lesion


def test_normal_marginals_are_dangerous_when_a_relative_uncertainty_is_large():
    # 57.6 % on a mass that enters as m^-0.96: a normal marginal puts probability near m = 0
    lognormal_cv = propagate(_lesion_model("liver lesion"), n=2**17, seed=1).summary().cv
    normal_cv = (
        propagate(_lesion_model("liver lesion", marginal="normal"), n=2**17, seed=1).summary().cv
    )
    assert normal_cv > 2 * lognormal_cv


_F = "gear2018.fit.liver."
GEAR_TAC_COV = np.array(  # Table 8, liver lesion
    [
        [_claim(_F + "var_a0"), _claim(_F + "cov_a0_rate")],
        [_claim(_F + "cov_a0_rate"), _claim(_F + "var_rate")],
    ]
)
GEAR_TAC_FIT = MonoExpFit(_claim(_F + "a0"), _claim(_F + "rate"), GEAR_TAC_COV, 3)  # Table 7


def _tia_model(correlated: bool) -> UncertainModel:
    params, corr = GEAR_TAC_FIT.to_parameters()
    if not correlated:
        params, corr = [Parameter(p.name, p.dist) for p in params], None
    return UncertainModel(
        "tia",
        InputSpace(params, corr),
        lambda x: x["fit_a0_mbq"] / x["fit_rate_per_h"],
        outputs="tia",
    )


def test_gear_tac_fit_uncertainty_needs_the_published_correlation():
    # delta method with the published covariance (Table 9 lists 15.1 % for the fit term)
    grad = np.array([1 / 0.026, -19.6 / 0.026**2])
    delta = math.sqrt(grad @ GEAR_TAC_COV @ grad) / (19.6 / 0.026)
    assert delta == pytest.approx(0.151, abs=0.004)
    mc = propagate(_tia_model(True), n=2**17, seed=2).summary().cv
    assert mc == pytest.approx(0.151, abs=0.015)
    independent = propagate(_tia_model(False), n=2**17, seed=2).summary().cv
    assert independent > 2.5 * mc  # ignoring r = 0.89 triples the uncertainty
