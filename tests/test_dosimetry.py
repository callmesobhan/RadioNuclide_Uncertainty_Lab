"""Kinetics, MIRD arithmetic and nuclide data."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from RadioNuclide_Uncertainty_Lab import InputSpace
from RadioNuclide_Uncertainty_Lab.kinetics import (
    LN2,
    MonoExpFit,
    effective_rate,
    fit_mono_exponential,
    mono_exponential_tia,
    multi_exponential_tia,
    time_activity,
    trapezoid_tia,
)
from RadioNuclide_Uncertainty_Lab.mird import MEV_TO_J, absorbed_dose_gy, self_s_value
from RadioNuclide_Uncertainty_Lab.nuclides import NUCLIDES, get_nuclide
from RadioNuclide_Uncertainty_Lab.sampling import unit_samples


def test_effective_rate_combines_physical_and_biological():
    assert effective_rate(10.0, 10.0) == pytest.approx(2 * LN2 / 10.0)
    # T_eff = Tp*Tb/(Tp+Tb)
    assert LN2 / effective_rate(159.53, 72.0) == pytest.approx(159.53 * 72 / (159.53 + 72))


def test_mono_exponential_tia_matches_numerical_integral():
    a0, k = 120.0, 0.0137
    numeric, _ = quad(lambda t: a0 * math.exp(-k * t), 0, math.inf)
    assert mono_exponential_tia(a0, k) == pytest.approx(numeric, rel=1e-8)


def test_multi_exponential_with_uptake_phase():
    amps, rates = np.array([100.0, -100.0]), np.array([0.02, 0.5])
    numeric, _ = quad(lambda t: float(time_activity(np.array(t), amps, rates)), 0, math.inf)
    assert multi_exponential_tia(amps, rates) == pytest.approx(numeric, rel=1e-6)


def test_trapezoid_tia_close_to_analytic_on_dense_grid():
    a0, k, tp = 100.0, 0.02, 159.53
    t = np.linspace(0, 120, 400)
    # tail assumes physical decay only, which *over*estimates a faster true clearance
    est = trapezoid_tia(t, a0 * np.exp(-k * t), tp)
    exact_body = a0 / k * (1 - math.exp(-k * 120))
    tail = a0 * math.exp(-k * 120) / (LN2 / tp)
    assert est == pytest.approx(exact_body + tail, rel=1e-3)
    assert est > mono_exponential_tia(a0, k)


def test_trapezoid_rejects_unsorted_times():
    with pytest.raises(ValueError):
        trapezoid_tia(np.array([0.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]), 100.0)


def test_fit_recovers_parameters_within_reported_uncertainty():
    rng = np.random.default_rng(11)
    t = np.array([4.0, 24.0, 48.0, 96.0, 168.0])
    truth_a0, truth_k = 240.0, 0.018
    sigma = 0.06 * truth_a0 * np.exp(-truth_k * t)
    a = truth_a0 * np.exp(-truth_k * t) + rng.normal(0, sigma)
    fit = fit_mono_exponential(t, a, sigma)
    sd = np.sqrt(np.diag(fit.cov))
    assert abs(fit.a0_mbq - truth_a0) < 4 * sd[0]
    assert abs(fit.rate_per_h - truth_k) < 4 * sd[1]


def test_fit_to_published_data_is_close_to_published_fit():
    # Gear 2018 Table 6 (liver lesion, 90Y activity from 111In SPECT) and Table 7 (their fit:
    # A0 = 19.6 MBq, lambda = 0.026 /h). Their weighting is not stated; a plain fit with the
    # same fractional uncertainty per point should land close.
    t = np.array([19.7, 45.1, 66.5])
    a = np.array([13.1, 5.3, 4.0])
    fit = fit_mono_exponential(t, a, 0.221 * a)
    assert fit.a0_mbq == pytest.approx(19.6, rel=0.15)
    assert fit.rate_per_h == pytest.approx(0.026, rel=0.10)


def test_fit_parameters_are_one_correlated_group():
    t = np.array([4.0, 24.0, 48.0, 96.0, 168.0])
    a = 240 * np.exp(-0.018 * t) * np.array([1.02, 0.97, 1.01, 0.98, 1.03])
    params, corr = fit_mono_exponential(t, a, 0.05 * a).to_parameters()
    assert {p.group for p in params} == {"biokinetic_fit"}
    (rho,) = corr.values()
    assert 0 < rho < 1  # amplitude and rate trade off positively for a decaying curve


def test_lognormal_fit_parameters_reproduce_the_fit_covariance():
    cov = np.array([[26.1, 3.77e-2], [3.77e-2, 6.91e-5]])  # Gear 2018 Table 8 (liver lesion)
    params, corr = MonoExpFit(19.6, 0.026, cov, 3).to_parameters()
    x = InputSpace(params, corr).transform(unit_samples(2**17, 2, seed=5))
    a0, rate = x["fit_a0_mbq"], x["fit_rate_per_h"]
    assert a0.mean() == pytest.approx(19.6, rel=0.01)
    assert a0.std() == pytest.approx(5.10, rel=0.02)
    assert rate.std() == pytest.approx(0.00831, rel=0.03)
    pearson = np.corrcoef(a0, rate)[0, 1]
    assert pearson == pytest.approx(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]), abs=0.01)
    assert a0.min() > 0 and rate.min() > 0


def test_fit_to_parameters_rejects_unknown_marginal():
    fit = MonoExpFit(10.0, 0.02, np.diag([1.0, 1e-6]), 4)
    with pytest.raises(ValueError, match="marginal"):
        fit.to_parameters(marginal="gamma")


def test_self_s_value_hand_calculation():
    # 1 MeV absorbed fully per decay in 1 kg: 1.602e-13 Gy/decay * 3.6e9 decays/(MBq h)
    assert self_s_value(1.0, 1.0, 1.0) == pytest.approx(MEV_TO_J * 3.6e9)
    expected = 0.1479 * 1.602176634e-13 / 0.422 * 3.6e9
    assert self_s_value(0.1479, 1.0, 0.422) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "mass_kg, s_published, ratio_to_delta_e",
    [  # Sjogreen Gleisner 2022 (EANM), Table 2: IDAC-Dose 2.1, adult male, Gy/(MBq h)
        (0.422, 0.204e-3, 1.01),  # kidneys
        (2.360, 0.0376e-3, 1.04),  # liver
        (0.2284, 0.377e-3, 1.01),  # spleen
    ],
)
def test_electron_only_s_value_reproduces_published_self_dose_within_photon_share(
    mass_kg, s_published, ratio_to_delta_e
):
    s_electron = self_s_value(get_nuclide("Lu-177").delta_e_mev, 1.0, mass_kg)
    # the published table reports how far the full (electron + photon) S exceeds the
    # electron-only estimate; reproduce that ratio
    assert s_published / s_electron == pytest.approx(ratio_to_delta_e, abs=0.015)


def test_absorbed_dose_is_product():
    assert absorbed_dose_gy(1000.0, 2.5e-4) == pytest.approx(0.25)


def test_nuclide_registry_only_holds_sourced_entries():
    lu = get_nuclide("Lu-177")
    assert lu.half_life_h == pytest.approx(6.6443 * 24)  # current DDEP, not the older 6.647 d
    assert lu.half_life_unc_d == pytest.approx(0.0009)
    assert lu.delta_e_mev == pytest.approx(0.1479)
    assert "ICRP" in lu.source and "DDEP" in lu.source and "Mougeot" in lu.source
    assert sorted(NUCLIDES) == ["Lu-177"]  # no unverified placeholder entries
    with pytest.raises(KeyError, match="unknown nuclide"):
        get_nuclide("Y-90")
