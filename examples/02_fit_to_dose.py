"""From a time-activity fit to a lesion dose, with the fit's correlation kept.

Uses the worked example of Gear et al. 2018 (EANM uncertainty guidance, open access): a liver
lesion treated with 90Y-DOTATATE and imaged with 111In SPECT at three times.

* Table 6: activities at 19.7, 45.1 and 66.5 h (MBq): 13.1, 5.3, 4.0
* Tables 7-8: published mono-exponential fit and covariance
* Tables 10-13: lesion S-value S = 0.429 * m^-0.961 Gy/(MBq h) (m in g), mass 13.9 g (57.6 %)

The amplitude and rate of a fitted exponential are strongly correlated (r = 0.89 here). Treating
them as independent triples the uncertainty of the time-integrated activity.

Run from the repository root:  python examples/02_fit_to_dose.py
"""

import numpy as np

from RadioNuclide_Uncertainty_Lab import InputSpace, Parameter, UncertainModel, lognormal, propagate
from RadioNuclide_Uncertainty_Lab.kinetics import MonoExpFit, fit_mono_exponential

# --- 1. fitting your own data -----------------------------------------------------------------
t = np.array([19.7, 45.1, 66.5])
activity = np.array([13.1, 5.3, 4.0])
mine = fit_mono_exponential(t, activity, sigma_mbq=0.221 * activity)
print(f"fit of Table 6 data:  A0 = {mine.a0_mbq:.1f} MBq, rate = {mine.rate_per_h:.4f} /h")

# --- 2. the published fit and covariance (Tables 7-8) -----------------------------------------
cov = np.array([[26.1, 3.77e-2], [3.77e-2, 6.91e-5]])
published = MonoExpFit(a0_mbq=19.6, rate_per_h=0.026, cov=cov, n_points=3)
print(
    f"published fit:        A0 = 19.6 MBq, rate = 0.0260 /h, "
    f"r = {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}\n"
)

fit_params, fit_corr = published.to_parameters()  # lognormal marginals, covariance preserved
mass = Parameter(
    "mass_g",
    lognormal(13.9, 1.58),
    group="lesion_volume",
    unit="g",
    source="Gear 2018 Table 1: 57.6 % (voxelisation + resolution)",
)


def dose(x):
    tia = x["fit_a0_mbq"] / x["fit_rate_per_h"]  # MBq h
    return tia * 0.429 * x["mass_g"] ** -0.961  # Gy


def run(label, params, corr):
    model = UncertainModel(label, InputSpace(params, corr), dose, outputs="dose")
    s = propagate(model, n=2**16, seed=1).summary()
    print(f"{label:<34} dose = {s.median:5.1f} Gy (median), CV {s.cv:6.1%}")


run("fit correlation kept (r = 0.89)", [*fit_params, mass], fit_corr)
independent = [Parameter(p.name, p.dist, p.unit) for p in fit_params]
run("fit parameters treated independent", [*independent, mass], None)
print(
    "\nNot comparable with Gear 2018 Table 13 (26.1 Gy, 37.6 %): their chain also carries the\n"
    "calibration / recovery / volume errors and correlates activity with lesion volume\n"
    "(r = -0.80), which lowers the dose uncertainty. That full reproduction is in\n"
    "tests/test_benchmarks.py; this example isolates the effect of the fit correlation."
)
