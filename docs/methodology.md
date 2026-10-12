# Methodology

## 1. Dose model (MIRD, organ level)

Absorbed dose to a target region is the sum over source regions of time-integrated activity times
an S-value:

    D_t = Σ_s  Ã_s · S(t←s),        Ã_s = ∫₀^∞ A_s(t) dt

`models.multi_organ` implements this for organs that are both source and target, with

    kinetics type mono_exponential:   Ã = f · A_admin · T_eff / ln2     (f = uptake fraction at t = 0)
    kinetics type tia_coefficient:    Ã = c · A_admin                   (c in hours)
    self S-value for organ mass m:    S(t←t) = S_ref · (m / m_ref)^(−p)
    cross S-values:                   S(t←s), s ≠ t: constants, from the same dataset as the self terms

`T_eff` is the *measured* effective half-life, so physical decay is already included. The mass
exponent is p = 1 for electron self-dose in an organ much larger than the electron range (same
absorbed energy, different mass); small lesions need other power laws (Gear et al. fit
S = 0.429·m^−0.961 Gy/(MBq h), m in g, for 90Y).

Multiplicative error factors, nominal 1, enter as

    Ã_s → Ã_s · Π activity_scale.* · Π s.activity_scale.*
    D_t → D_t · Π dose_scale.*     · Π t.dose_scale.*

where unprefixed names are **shared** by all organs and organ-prefixed names are **specific**.

With `s_values: <dataset>` an organ takes its self S-value and reference mass from a packaged
dataset by `region`, and the cross terms between all such organs are added from the same dataset, so
self and cross terms always share one phantom and one set of decay data. Cross-organ terms are used
at the reference mass; mass scaling applies to the self term only (cross-dose is ~2 % of the dose).

Units: MBq, h, kg, Gy. `mird.self_s_value` converts an electron energy per decay to Gy/(MBq h)
using 1 MeV = 1.602176634×10⁻¹³ J and 1 MBq h = 3.6×10⁹ decays; it is used to cross-check
tabulated S-values, which include photon self-dose.

## 2. Where input uncertainty comes from

The shipped scenario uses the groups of Staanum et al. 2021, Table 2 (kidney, 3 SPECT scans). They
are standard (1σ) relative uncertainties, combined by the authors in quadrature. Typical group
meaning and treatment:

| Group | Contents | Treatment here |
|---|---|---|
| `dose_calibrator` | Calibration and stability of the activity calibrator | shared; N(1, 1 %) each |
| `spect_quantification` | Scanner sensitivity, dead-time constant and correction, reproduction of activity, stability | shared; N(1, 1–6 %) |
| `delineation_pvc.<organ>` | VOI delineation, partial-volume correction | per organ; N(1, 3 %), N(1, 7 %) |
| `tac_model.<organ>` | Single-exponential extrapolation (early, late) | per organ; N(1, 1 %) each |
| `tac_fit.<organ>` | Curve-fit uncertainty | kidneys: correlated lognormal (C0, T_eff), r = −0.76; liver/spleen: N(1, 6.1 %) on the TIA |
| `dose_factor.<organ>` | Photon absorption, organ density | per organ; N(1, 1 %) each |

Type A (statistical, e.g. a fit covariance) and Type B (judgement, e.g. a manufacturer tolerance)
uncertainties are both expressed as distributions, following the approach of GUM Supplement 1.
Distributions of multiplicative factors with small relative SD are normal about 1; quantities with
large relative uncertainty (fit parameters, masses) are lognormal (§3).

**What this is not.** These are the uncertainties of one patient's *measurement chain*. The spread
of doses between patients (e.g. kidneys 0.43–0.65 Gy/GBq across cohorts) is variability, answers a
different question, and is not in the budget.

## 3. Monte Carlo propagation and the choice of marginal

Inputs are sampled on the unit hypercube, mapped through each marginal's inverse CDF, and pushed
through the model. Defaults to a scrambled Sobol' sequence (`n` rounded up to a power of two);
Latin hypercube and plain random sampling are available.

For each output, `Summary` gives mean, SD, median, a central percentile interval (valid for the
skewed outputs of multiplicative models), skewness and, for strictly positive outputs, the
geometric SD. `mean_se = SD/√n` is the iid formula, conservative for QMC/LHS.

**The marginal matters more than the propagation method at large uncertainty.** With first-order
(GUM-style) propagation the *shape* of the input distribution is irrelevant; with Monte Carlo it
is not. Reproducing Gear et al. 2018's liver lesion (57.6 % on the mass, entering as m^−0.96) with
a normal marginal truncated at zero gives a dose CV of ≈ 550 %, because the density near m → 0 is
not negligible; with a lognormal it gives 35.3 % against the published first-order 37.6 %. Use a
positive, right-skewed distribution whenever the relative SD exceeds roughly 20–30 %.

## 4. Correlation: Gaussian copula inside a group

For a correlated group, unit-cube columns `u` are mapped to normal scores `z = Φ⁻¹(u)`,
correlated with the Cholesky factor `L` of the requested matrix (`z' = L z`), mapped back
`u' = Φ(z')`, and finally through the marginal inverse CDFs.

`rho` is therefore the correlation of the **latent normal scores**. It equals the Pearson
correlation only for normal marginals; the Spearman correlation is `(6/π)·asin(ρ/2)` regardless of
marginals. For a fit with lognormal marginals, `MonoExpFit.to_parameters` solves

    cov(X, Y) = E[X] E[Y] (exp(ρ_z σ₁ σ₂) − 1),   σᵢ² = ln(1 + cvᵢ²)

for `ρ_z`, so the **fit covariance is reproduced exactly** (checked on Gear Table 8: Pearson
0.8877, SDs 5.11 and 0.00831 against the published 5.10 and 0.00831).

Why correlation inside a fit matters: for a decaying exponential, amplitude and rate trade off
(r = +0.89 in Gear's example), so errors in `A0/λ` partly cancel. Treating them as independent
raises the TIA uncertainty from 14 % to 42 %.

## 5. Sensitivity analysis

### Screening (`rank_sensitivity`)

Uses the existing Monte Carlo sample, no extra model runs: **Spearman** (rank correlation),
**PRCC** (partial rank correlation with the other inputs' ranks regressed out) and **SRC**
(standardised regression coefficient on raw values), plus the `R²` of that regression. If `R²` is
well below 1 the model is non-linear or interacting and SRC is not a variance share. These treat
parameters individually (even inside a correlated group) and assume a monotonic response.

### Variance-based indices (`sobol_all`, `sobol_indices`)

For output `Y = f(X)` with independent factors `X₁…X_K`:

- **First-order** `S_k = Var(E[Y | X_k]) / Var(Y)`: share explained by factor `k` alone.
- **Total-order** `ST_k`: share removed if all *other* factors were fixed, i.e. `S_k` plus every
  interaction containing `k`.

With matrices `A`, `B` of independent samples and `AB_k` = `A` with the columns of factor `k`
taken from `B`:

    S_k  ≈ mean( f(B) · (f(AB_k) − f(A)) ) / Var      Saltelli et al. (2010)
    ST_k ≈ ½ · mean( (f(A) − f(AB_k))² ) / Var        Jansen (1999)

Cost `n·(2+K)` evaluations, **once for all outputs**. Notes:

- `f(B)` is centred before use: the expectation is unchanged but a large variance term vanishes
  when |mean| ≫ SD, as for doses (first-order CIs tightened ≈ 4× at identical point estimates).
- A factor is a **group**; its columns are swapped together, so correlation inside the group is
  preserved and the group's index is well defined.
- Confidence intervals are percentile bootstraps over sample rows (iid assumption: conservative for
  a QMC design). Near-zero indices can be slightly negative (noise).
- `1 − ΣS_k` is the **interaction share**, reported rather than renormalised away.
- Zero-variance outputs (e.g. a ratio when only shared factors are uncertain) raise an error naming
  the output.

## 6. Error budget and the factor × output matrix

Per output and factor: `S1`, `ST` (with CIs) and the equivalent standard deviation
`√(share · Var(Y))` and its ratio to the output mean. Shares add (`ΣS1 ≤ 1`); **SDs do not**, so the
budget leads with the share. `budget_matrix` lays total-order shares out as factors × outputs.

How to read the matrix of the shipped scenario:

- A factor lit in **every organ column** (`spect_quantification`, `dose_calibrator`) is a *shared*
  systematic error: it correlates the organ doses and is absent from their ratios.
- A factor lit in **one organ column** is organ-specific.
- **Ratio columns** keep only organ-specific factors, twice (for numerator and denominator), which
  is why a ratio can be less certain than either dose.

For a product model, shares are close to the normalised squared *log*-uncertainties; Staanum's
budget is exactly that (shares = squared relative SD / total), and the Sobol' result matches it
within the sampling error.

## 7. Limitations

- **Parametric uncertainty only.** Choosing the wrong curve model or phantom is *structural*
  uncertainty that variance-based methods on one model cannot see. Compare models in separate
  configs and treat model choice as a scenario.
- **Measurement chain, not variability** (§2).
- **Factors are independent of each other.** A shared error that cannot be written as one
  multiplicative factor needs dependence between groups; see the roadmap.
- **Cross-organ dose** uses phantom (reference-mass) S-values; patient-specific cross-dose is not
  modelled. Remainder of body, tumours and red marrow kinetics are not in the shipped scenario.
- **Transferred budgets.** Liver and spleen reuse the kidney budget as a stated assumption.
- **QMC conventions**: rounded sample sizes, conservative error bars.

## 8. Anchoring and data checks

**Anchoring.** To make a reference scenario reproduce published organ doses, the time-integrated
activity coefficients are not chosen organ by organ but solved jointly from `S · T = D`
(`mird.tia_coefficients_from_doses`), where `S[t, s]` is the self and cross S-value matrix and
`D[t]` the published *total* dose per activity (Gy/GBq equals mGy/MBq). With cross terms present,
dividing each dose by its own self S-value would overshoot by the cross share (1.8 % for the kidneys).
The solver rejects a non-square system and non-positive coefficients, which signal inconsistent
doses.

**Physics checks on an S-value dataset.** (a) *Reciprocity*: specific absorbed fractions are symmetric,
so `S(t←s) = S(s←t)` up to Monte Carlo noise (0.2 % among kidneys, liver and spleen; up to 4.7 % for
marrow pairs). The noise grows as a cross term shrinks, so only cross terms at least 10⁻³ of the geometric
mean of the two self terms are tested (asymmetry reaches 13 % at 10⁻⁵ and is irrelevant to any dose). (b) *Energy conservation*: for each source, `Σ_t S(t←s) · m_t` over the
listed targets cannot exceed the energy emitted per MBq h, `(Δe + Δph) · 1.602×10⁻¹³ J/MeV · 3.6×10⁹`
(the largest ratio is 0.85, since only four of ~48 targets are listed). (c) *Cross-database agreement*:
self S-values and masses against an independent code (MIRDcalc vs IDAC-Dose, ≤ 0.4 %), and **cross terms** for a
liver source in the female phantom against IDAC-Dose 2.1 (≤ 0.40 %) and OpenDose (≤ 0.52 %). (d) *Primary nuclear
data*: the mean electron energy per decay must lie between the DDEP β⁻ averages plus the minimum, and plus
the maximum, conversion- and Auger-electron energy of each group, [140.7, 151.0] keV; ICRP 107 (147.9) and NNDC
(146.5) both do.

**Claim checks.** Quadrature totals, delta-method uncertainties, nominal doses from published
inputs and the like are written as expressions over claims and compared within a tolerance, so a
source's own arithmetic is audited, not just copied (see `docs/data-sources.md` §6 for the
discrepancies this found).

## References

- JCGM 101:2008, *Evaluation of measurement data – Supplement 1 to the GUM – Propagation of
  distributions using a Monte Carlo method.*
- Bolch WE et al., *MIRD Pamphlet No. 21: A generalized schema for radiopharmaceutical dosimetry –
  standardization of nomenclature*, J Nucl Med 2009;50:477–484.
- Gear JI et al., *EANM practical guidance on uncertainty analysis for molecular radiotherapy
  absorbed dose calculations*, Eur J Nucl Med Mol Imaging 2018;45:2456–2474.
- Staanum et al., *Practical kidney dosimetry in peptide receptor radionuclide therapy using
  [177Lu]Lu-DOTATOC and [177Lu]Lu-DOTATATE with focus on uncertainty estimates*, EJNMMI Phys
  2021;8:78.
- Sjögreen Gleisner K et al., *EANM dosimetry committee recommendations for dosimetry of
  177Lu-labelled somatostatin-receptor- and PSMA-targeting ligands*, Eur J Nucl Med Mol Imaging
  2022;49:1778–1809.
- Sobol' IM, *Global sensitivity indices for nonlinear mathematical models and their Monte Carlo
  estimates*, Math Comput Simul 2001;55:271–280.
- Jansen MJW, *Analysis of variance designs for model output*, Comput Phys Commun 1999;117:35–43.
- Saltelli A et al., *Variance based sensitivity analysis of model output. Design and estimator for
  the total sensitivity index*, Comput Phys Commun 2010;181:259–270.
- Kesner AL et al., *MIRD Pamphlet No. 28, Part 1: MIRDcalc*, J Nucl Med 2023;64(10):1524–1531
  (source of the S-value database used here).

Please check citation details against the originals before reusing them in a paper.
