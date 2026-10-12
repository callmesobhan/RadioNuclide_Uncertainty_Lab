# Roadmap

Ordered by value to a dosimetry analyst; each item says what test would prove it works.

## Done in 0.2

- Multi-organ model with self and cross S-values, shared and organ-specific error factors, mass
  scaling, ratio outputs; multi-output propagation, Sobol' and budget matrix.
- Shipped scenario rebuilt from published data; published results reproduced.
- **Cross-organ S-values** from the MIRDcalc database (validated dataset with provenance and a
  regeneration tool); anchoring solved jointly.
- **Data robustness**: claims registry, consistency checks with automatic exclusion, known-conflict
  tracking, dataset gate, derivation tests, run-time diagnostics; independent second-source validation of the
  cross terms (female phantom) and primary-source nuclear data.

## Next: close the data gaps

- **Red marrow** (blood-based and image-based): the S-values are already in the dataset, the
  kinetics are not. *Test:* reproduce a published marrow dose and its uncertainty.
- **Tumours**: the MIRDcalc file holds Lu-177 sphere S-values (radius, mass, photon/beta/total) that
  would replace the Y-90 power law. *Test:* reproduce a published tumour dose and uncertainty.
- **A second source for the *male* cross terms.** The cross terms are validated against IDAC-Dose and
  OpenDose on the female phantom (0.4–0.5 %); for the male phantom they rest on reciprocity, energy
  conservation and the EANM's "about 2 %". OpenDose's website (unreachable here) or ICRP 133 SAFs with ICRP
  107 data would close this.
- **Phantom-selection key** for the scenario config (the female dataset is already packaged; the CSVs
  exist for 12 phantoms including paediatric).
- **Claims at scale**: a helper that proposes claim entries from a pasted table so transcription
  mistakes are caught at entry time.
- **Organ-specific measurement budgets** for liver and spleen to replace the kidney-derived
  assumption. *Test:* budget total within the cited paper's.
- **More nuclides** (Y-90, I-131) with primary DDEP evaluations; Lu-177 is done (T½ and emission data from
  LNHB DDEP). Add the claims, bounds checks and the registry-vs-claims check for each.

## Then: kinetics that match real data

- **Bi-exponential and uptake-phase models** (`A(t) = Σ Aₖ e^{−λₖ t}`) as a kinetics type plus
  `fit_multi_exponential`. *Test:* analytic TIA vs `quad`; fit recovery.
- **Single-time-point dosimetry** (fixed effective half-life): the 1-SPECT path of the Staanum
  budget (19 %). *Test:* reproduce 19 %.
- **Sampling-schedule analysis**: dose uncertainty each candidate imaging schedule would give.
  *Test:* more late points never increase fit-driven variance on synthetic data.
- **Convergence diagnostics**: replicated randomised-QMC estimates and an "n needed for ±x %"
  helper, replacing the conservative iid standard error.

## Later

- **Remainder of body**, whole-body dose output, and **cumulative dose over cycles** with
  cycle-to-cycle variability.
- **Dependence between groups** (shared latent variables, Shapley effects) for cases where a
  shared error cannot be written as one multiplicative factor.
- Bayesian fit (posterior samples as inputs); biological-effective-dose output.
- Benchmarks from further EANM guidance examples; golden-file regression tests; docs site;
  PyPI release.

## Explicitly out of scope

Radiation transport, image reconstruction/segmentation, and anything that presents results as
clinically validated.
