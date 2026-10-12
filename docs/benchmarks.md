# Benchmarks

Two kinds of checks run in CI ([`tests/test_benchmarks.py`](../tests/test_benchmarks.py),
[`tests/test_uncertainty.py`](../tests/test_uncertainty.py), [`tests/test_dosimetry.py`](../tests/test_dosimetry.py)):
published results reproduced from their own inputs, and closed-form answers. All use fixed seeds,
so they are deterministic.

## Published results

| # | Quantity | Published | Reproduced | Tolerance in test |
|---|---|---|---|---|
| 1 | Kidney dose uncertainty, 3 SPECT scans [Staanum2021 Table 2] | 13 % (each term rounded up) | quadrature of the listed terms 12.13 %; Monte Carlo 12.0 % | ±0.4 pp |
| 2 | Sobol' variance share per Staanum source group (kidneys) | quadrature shares (delineation+PVC 39.4 %, SPECT 31.2 %, fit 25.3 %, others 1.4 % each) | Sobol' ST 39.4 / 31.4 / 25.2 / 1.4 % | ±3 pp |
| 3 | Liver lesion: Ã = 762 MBq h, m = 13.9 g [Gear2018 Tables 9–13] | D = 26.1 Gy | 26.06 Gy | 0.5 % |
| 4 | Liver lesion first-order u(D)/D from u(Ã)=26.7 %, u(m)=57.6 %, r(Ã,S)=−0.80 | 37.6 % | 37.58 % | ±0.3 pp |
| 5 | Pancreatic lesion: Ã = 5933 MBq h, m = 142 g | D = 21.7 Gy, u = 15.6 % | 21.75 Gy, 15.58 % | 0.5 %, ±0.3 pp |
| 6 | Monte Carlo vs first-order, same inputs scaled ×0.05 (linear regime) | exact agreement expected | agree to 4 digits (1.88 %) | 2 % relative |
| 7 | Same lesions, full uncertainties, lognormal marginals | 37.6 % / 15.6 % (first order) | 35.3 % / 15.3 % | 10 % / 5 % relative |
| 8 | Same, normal marginals truncated at 0 | – | CV ≈ 550 % for the liver lesion | > 2× the lognormal result |
| 9 | Fit covariance [Gear2018 Table 8], r(A0, λ) | 0.89 | sample Pearson 0.8877; SDs 5.11 and 0.00831 vs 5.10 and 0.00831 | ±0.01 |
| 10 | TIA fit term, 15.1 % [Gear2018 Table 9] | 15.1 % | delta method 14.9 %; Monte Carlo 14.3 %; independent parameters 42 % | ±1.5 pp; independent > 2.5× |
| 11 | Fit of the three Table 6 points vs the published fit (A0 = 19.6 MBq, λ = 0.026 /h) | 19.6, 0.026 | 19.7, 0.0258 | 15 %, 10 % |
| 12 | Self S-values [Gleisner2022 Table 2] / electron-only estimate | 1.01, 1.04, 1.01 (their "ratio to Δe") | 1.009, 1.041, 1.009 | ±0.015 |
| 13 | Nominal scenario doses vs cohort means (0.46 / 0.54 / 0.61 Gy/GBq), self + cross dose | by construction (data-sources §3) | 0.460 / 0.540 / 0.610 | 0.2 % |
| 14 | MIRDcalc self S-values and masses vs IDAC-Dose 2.1 [Gleisner2022 Table 2] | 0.204 / 0.0376 / 0.377 / 0.0349 mGy/MBq/h; 422 / 2360 / 228.4 / 1394 g | MIRDcalc within 0.38 %; masses identical | ±1 % |
| 15 | Rest-of-body share of the kidney dose [Gleisner2022] | "about 2 %" | 1.8 % from the cross-organ S-values | ±0.7 pp |
| 16 | `S · T = D` anchoring | reproduces D exactly | residual < 1e-12 | exact |
| 17 | Source discrepancies detected by the checks (six, see data-sources §6) | – | all six fail as expected, nine claims excluded | exact set |
| 18 | Lu-177 S-values, liver source, ICRP 110 female [OpenDose2020 Table 1]: six targets, 8.4e-10 to 1.4e-5 mGy/(MBq s) | IDAC-Dose 2.1 and OpenDose | MIRDcalc within **0.40 %** of IDAC, **0.52 %** of OpenDose (liver→spleen 0.09 % / 0.37 %) | ±1 % |
| 19 | Mean electron energy per decay | ICRP 107 147.9 keV; NNDC 146.5 keV | both inside the DDEP-derived interval [140.7, 151.0] keV | strict bounds |
| 20 | Lu-177 half-life | DDEP (current) 6.6443(9) d; Staanum 6.647 d | 0.04 % apart; registry equals the DDEP claim | exact / ±0.1 % |

Rows 7–8 are a methodological finding as much as a check: with a 58 % relative uncertainty on a
mass entering as m^−0.96, **the choice of marginal matters more than the choice of propagation
method**. A normal distribution puts probability near zero mass and the dose diverges. Use
lognormal (or another positive distribution) for large relative uncertainties; the package's
`MonoExpFit.to_parameters()` does this by default and preserves the published covariance exactly.

## Multi-organ structure

| Property | Prediction from the published budget | Result |
|---|---|---|
| Ratio uncertainty (organ-specific terms only, twice) | √2 × 9.96 % = 14.1 % | 14.0 % |
| Correlation between two organ doses (shared share of variance) | 48 / 147.2 = 0.33 | 0.33–0.34 |
| Shared factors in a ratio's Sobol' budget | exactly zero | below detection |

## Checks on the data itself (`check-data`)

Run by `radionuclide-uncertainty-lab check-data`, in CI and in `tests/test_datacheck.py`: 40 checks
over 131 claims; 34 pass and 6 are declared known conflicts of the sources. Beyond the
published-result rows above:

| Check | What it verifies | Result |
|---|---|---|
| `svalues.reciprocity` (male, female) | S(t←s) = S(s←t) for cross terms ≥ 1e-3 of the self terms (smaller ones are Monte Carlo noise: up to 13 % at 1e-5) | worst 4.7 % male / 3.4 % female (kidneys–marrow) |
| `svalues.reciprocity_scenario_organs` | the same, for kidneys, liver, spleen only, tolerance 1 % | 0.20 % (liver–spleen) |
| `svalues.male.kidney_cross_share_vs_eanm` | share of the anchored kidney dose from liver and spleen vs "about 2 %" | 1.80 % (±0.7 pp) |
| `svalues.female.cross_vs_idac` / `_vs_opendose` | MIRDcalc cross terms vs two independent codes | 0.40 % / 0.52 % |
| `nuclide.delta_e_within_ddep_bounds` | ICRP 107 and NNDC electron energy vs bounds from DDEP | inside [140.7, 151.0] keV |
| `nuclides.registry_matches_claims` | `nuclides.py` equals the claims it cites | exact |
| `svalues.energy_conservation` (male, female) | absorbed ≤ emitted energy per source | worst 0.85 (liver) |
| `svalues.agree_with_idac_self_s` | MIRDcalc vs IDAC-Dose self S and masses | worst 0.38 % |
| `staanum.quadrature_3spect` | Table 2 terms add to the printed total | 12.13 % vs 13 % (rounded up) |
| `gear.*` (8 passing checks) | S-model, nominal dose, first-order uncertainty, fit-covariance arithmetic | all within 0.5–3 % |
| `kidney_dose.*` | two cohorts agree; cohort spread bounded | 0.46 vs 0.43 (7 %); max/min 1.51 |

Mutation tests prove the checks can fail: doubling one cross term trips reciprocity (but a 5× error in a
noise-dominated term does not, and a pair set with nothing above the floor fails closed), scaling a self
value by 10 trips energy conservation, scaling spleen self-S by 10 % trips the IDAC cross-database check, a 3 %
error in liver→spleen trips the strict scenario-organ reciprocity check and the female cross-term checks, a
tenfold kidney cross term trips the EANM share check, a 165 keV electron energy trips the DDEP bound, a
stale half-life in `nuclides.py` trips the registry check, and a dataset whose checks fail (or whose declared
known conflict has silently disappeared) is refused by `multi_organ`.

## Closed-form checks

| Test | Oracle |
|---|---|
| Ishigami function | analytic S1 = (0.314, 0.442, 0), ST = (0.558, 0.442, 0.244) |
| Product of three lognormals | analytic S1, ST (interactions are real, so ΣS1 < 1) |
| Additive model with a correlated group | group share (a² + b² + 2abρ)/Σ; no interaction |
| Gaussian copula | Spearman = (6/π)·asin(ρ/2); Pearson = ρ for normal marginals |
| Mono-exponential and bi-exponential TIA | numerical integral (`scipy.integrate.quad`) |
| Multi-output model | y1 = 2a + b, y2 = b + 3c: analytic shares and output correlation 1/√50 |
| MIRD arithmetic | hand calculation, 1 MeV·kg⁻¹·decay⁻¹ identity |
