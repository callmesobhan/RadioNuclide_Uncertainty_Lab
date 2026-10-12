# Design

## Goals

1. **Transparent**: every uncertain input is declared with a distribution *and a source*; no
   hidden defaults change the physics.
2. **Reproducible**: one seed and one config file determine the result; the report records both,
   plus a hash of the config and the library versions.
3. **Physically meaningful**: sensitivity results are attributed to *physical sources of
   uncertainty* (quantification, biokinetics, dose factor, …), not just to individual symbols, and
   shared systematic errors are distinguished from organ-specific ones.
4. **Small and checkable**: NumPy/SciPy/PyYAML only (matplotlib optional). Every estimator is
   tested against a published result or a closed-form answer ([benchmarks.md](benchmarks.md)).

Non-goals: being a general dosimetry package (no radiation transport, no image processing), and
clinical decision support.

## Architecture

Dependencies point downward; nothing in the lower layers knows about config files or reports.

```
 cli ─► workflow ─► (propagation, sensitivity, budget, diagnostics)
  │        │
  └─► config ─► models ─► (kinetics, mird, nuclides, refdata)
        │  │      │
        │  └──────┴──► datacheck ─► refdata, claims.yaml   (claims, checks, exclusion)
        └─────────────► model ─► inputs ─► distributions
 report, plots ─► Analysis (the result of workflow.analyze)
 tools/extract_mirdcalc_svalues.py ─► data/*.json           (offline, stdlib only)
```

| Layer | Modules | Responsibility |
|---|---|---|
| Statistics core | `distributions`, `inputs`, `sampling`, `model` | Describe uncertain inputs; turn unit-cube samples into physical values; wrap a vectorised function with one or more outputs. Domain-agnostic. |
| Analysis | `propagation`, `sensitivity`, `budget`, `workflow` | Monte Carlo summaries, screening, Sobol', budgets, factor × output matrix. Domain-agnostic. |
| Dosimetry | `kinetics`, `mird`, `nuclides`, `models` | Time-integrated activity, S-values, anchoring, the multi-organ dose model. |
| Data integrity | `refdata`, `datacheck`, `diagnostics` | Packaged datasets with provenance; published-number registry and consistency checks that exclude conflicting data; run-time warnings. |
| Interface | `config`, `cli`, `report`, `plots` | YAML in; files out. |

The split matters: the statistics core is tested on textbook functions (Ishigami, products of
lognormals, a linear multi-output model) with no dosimetry in the loop; the dosimetry layer is
tested against hand calculations and published tables with no statistics in the loop.

## Decisions and their reasons

**Model = pure, vectorised function of a dict of arrays, with named outputs.** `func({name:
array(n,)})` returns an array, an `(n, m)` array or a `{output: array}` mapping. Required for
Sobol' (10⁵ evaluations) and makes models trivially testable. All outputs come from the same input
samples, so their *joint* behaviour (organ-dose correlation, ratios) is preserved. Constants live
in the closure; only quantities with an uncertainty are inputs.

**Factors (groups), not parameters, are the unit of sensitivity.** Parameters that share
`group` are varied as one block in the Sobol' design. Reasons: (a) budget lines match the questions
an analyst can act on ("improve scanner calibration"), (b) correlated parameters cannot be
separated meaningfully by variance decomposition, so we do not pretend to, (c) a clean rule:
factors are independent, correlation lives inside factors.

**Shared vs organ-specific errors by name.** In `models.multi_organ`, `activity_scale.<label>`
multiplies every source organ's TIA, `<organ>.activity_scale.<label>` one organ's; `dose_scale.…`
likewise for the dose. A shared error is therefore *one parameter*, not several correlated ones,
which is exact and cheap, and the budget shows it as one factor appearing in every organ's column.
The consequence, verified in the benchmarks, is that shared factors cancel in dose ratios.

**Two kinetics types, because the literature reports two.** Papers give either a fitted curve
(uptake and effective half-life) or an organ TIA coefficient in hours. The model accepts both
(`mono_exponential`, `tia_coefficient`) rather than forcing a conversion. Physical decay is part
of the *measured* effective half-life, so no biological half-life or radionuclide is needed.

**S-values are inputs, not computed.** Self S-values come from a cited phantom calculation and are
rescaled for organ mass as `S·(m/m_ref)^−p` (p = 1 for electron self-dose in a large organ; lesions
follow other power laws, e.g. Gear's p = 0.961). Cross-organ S-values are optional constants. The
electron-only estimate in `mird.self_s_value` is kept as a cross-check, not as the engine.

**Correlation only inside a group, via a Gaussian copula.** Keeps marginals exactly as declared
while imposing dependence. The Cholesky factor is checked at construction. `rho` is the
latent-normal correlation; `MonoExpFit.to_parameters` converts a fit covariance into the latent
correlation that reproduces it *exactly* for lognormal marginals.

**Lognormal by default for fit parameters.** The Gear benchmark showed that normal marginals with
~30–60 % relative uncertainty give nonsense tails (CV 550 % where the published value is 38 %).

**Sobol' evaluates the design once for all outputs.** The cost is `n·(2+K)` model evaluations
regardless of the number of outputs; `sobol_all` returns every output, `sobol_indices(output=…)`
one of them.

**Two sensitivity tiers.** `rank_sensitivity` is free but per-parameter and only trustworthy for
monotonic models; `sobol_all` is the authoritative decomposition. The report prints the linear R²
so the reader can see when SRC should not be trusted.

**Quasi-random sampling by default** (scrambled Sobol'): sample sizes round up to powers of two,
documented and visible in `result.n`.

**Strict configs.** Unknown keys, unknown distribution parameters, a quantity that is both
uncertain and fixed, a misspelled organ or scale-factor kind, missing quantities: all errors. A
typo that silently changes a distribution is the worst failure mode for this kind of tool. (The
label after `activity_scale.` is free-form by design, so a misspelled *label* is a new factor.)

**Data policy.** Shipped numbers must trace to a cited source, be derived by code that a test
recomputes, or be a stated assumption ([data-sources.md](data-sources.md)). The nuclide registry holds
only sourced entries. See *Robustness layers* below for how this is enforced.

**Units.** MBq, h, kg, Gy, MeV, with the unit in the parameter name where it helps.
Conversions live in one place (`mird.py`).

## Robustness layers

Each layer answers one way a dosimetry uncertainty analysis goes wrong silently. They are
independent, so a failure in one does not depend on the others.

| Layer | Failure it prevents | Mechanism |
|---|---|---|
| **Claims** (`claims.yaml`) | A number copied wrongly, or with no trace of where it came from | Every published number is one entry with its source and location; configs cite entries with `claim:` and the loader verifies that the distribution *realises* the claim (relative SD within 3 %, correlation within 0.01), so a typo or an edit that drifts from the source is an error |
| **Checks** | A source that contradicts itself; two sources that disagree | `agree`, `quadrature`, `bound`, `spread` and `refdata` checks over claims, arithmetic and datasets, run by `check-data`, in CI and in the tests |
| **Exclusion** | Using data that failed a check | Claims named by a failed check are excluded: `require()` raises `ExcludedClaimError`; a check that cannot be evaluated fails *closed* and excludes what it guards |
| **Known conflicts** | A source error being forgotten, or silently fixed | `expect: conflict` keeps the check failing on purpose (claims excluded); if the conflict disappears the check reports `UNEXPECTED_PASS` and CI fails until someone reviews it |
| **Dataset gate** | Using a reference dataset that has not been verified | A dataset needs generator provenance, load-time structure checks (finite, non-negative, units, masses, photon <= total) and at least one registered check; `multi_organ` refuses it if any check fails. A *declared* known conflict with an external, already-excluded claim does not block; a known conflict that has silently disappeared does |
| **Derivation tests** | Hand-computed nominal values drifting from their inputs | Derived numbers (TIA coefficients, kidney uptake fraction) are recomputed from the dataset and claims in `tests/test_data_integrity.py` |
| **Diagnostics** | A plausible-looking but unreliable result | Rules on every analysis: missing seed, small samples, normal marginal reaching zero, heavy tail, wide Sobol' CI, inconsistent indices, strong interactions; they travel in `result.json` and `report.md` |
| **Provenance** | Not knowing what produced a result | Config hash, seed, versions, claims relied on, excluded claims, dataset fingerprint (source-file SHA-256 prefix) |

Design choices worth stating:

- *Exclude both sides of a conflict.* When two readings disagree the code cannot know which is right,
  so neither is usable until a person resolves it (edit the claim, or change the check to say which
  reading wins).
- *Checks are data, not code.* They live in YAML next to the claims, with a note saying what they
  mean, so a reviewer can read them without reading Python. Expressions use a whitelisted AST
  evaluator (numbers, claim names, `+ - * / **`, `sqrt abs log`), never `eval`.
- *Physics checks outrank source agreement.* Reciprocity and energy conservation need no second
  source, so they still protect a dataset when only one database is available.
- *Validate the pipeline where an independent source exists.* The male cross terms have no published
  independent table, but the same MIRDcalc pipeline is compared with IDAC-Dose and OpenDose on the female
  phantom; the check is registered for that dataset and the evidence is stated as covering the pipeline,
  not the male numbers themselves.
- *Noise-aware, not tolerance-tuned.* Reciprocity asymmetry grows as a cross term shrinks (Monte Carlo
  noise), so the check tests only pairs above a stated fraction of the self terms and reports how many it
  skipped; a stricter check covers exactly the organs the scenario uses. If nothing is testable it fails closed.
- *Registry-in-code is data too.* `nuclides.py` is tied to its claims by a check, which is what would have
  caught a stale half-life.
- *Diagnostics never change numbers.* They only describe how far to trust them.

## Extension points

| To add… | Do this |
|---|---|
| A distribution | Add a constructor in `distributions.py` and an entry in `_SPEC_KEYS`/`from_spec`; or pass any frozen scipy distribution to `Parameter`. |
| A dose model | Write a builder like `models.multi_organ`, register it in `MODEL_BUILDERS` with its extra config keys; or build an `UncertainModel` directly (see `examples/02_fit_to_dose.py`). |
| A lesion or tumour | Add an organ with `mass_exponent` ≠ 1 and the lesion's S at a reference mass (benchmarked on Gear's S = 0.429·m^−0.961). |
| Cross-organ dose | Give `cross_s_gy_per_mbq_h: {target: {source: S}}` from OpenDose / IDAC-Dose / MIRDcalc. |
| A nuclide | Add a `Nuclide` with a cited source; add the citation to `data-sources.md`. |
| A claim | Add it to `claims.yaml` with source and location, and at least one check that ties it to something else. |
| A reference dataset | Write (or reuse) a generator in `tools/`, emit JSON with provenance into `data/`, register checks with `dataset: <name>`; reference it from a config with `s_values: <name>`. |
| A check kind | Add a branch to `ClaimRegistry._evaluate` (or a function in `_REFDATA_CHECKS`) and a test showing it passing, failing and excluding. |
| A diagnostic | Add a rule to `diagnostics.diagnose` with a test that triggers it and one that does not. |
| A report format | Consume `report.to_dict(analysis)`; it is the stable machine interface. |

## Testing strategy

| Layer | Oracle |
|---|---|
| Distributions, copula | Median/GSD recovered; Spearman = 6/π·asin(ρ/2); Pearson = ρ |
| Kinetics | `quad` integral; fit recovers truth; published Gear fit and covariance reproduced |
| MIRD, S-values | Hand arithmetic; Gleisner Table 2 within its stated photon ratio |
| Propagation, Sobol' | Product of lognormals; Ishigami; additive model with correlated group; linear multi-output model |
| Multi-organ model | Hand calculation; mass scaling; cross-dose; shared vs specific factors; ratio cancellation |
| Published results | Staanum budget, Gear lesions and fit (see benchmarks.md) |
| Claims and checks | Each check kind passing, failing, excluding; known-conflict semantics; fail-closed; safe expression evaluator rejects non-arithmetic; the shipped registry has exactly the expected known conflicts |
| Datasets | Load-time rejection of every corruption; checks fail on a doctored matrix (asymmetric, impossible energy, wrong self value); untrusted dataset refused |
| Data integrity | Config anchors equal the derived values; every scenario parameter cites a claim; drifted or excluded claims refused; cross-dose switch changes doses by the expected 1.8 % |
| Diagnostics | Each rule fires on the situation it was written for; the shipped scenario is quiet; numbers unchanged |
| CLI / config | End-to-end; determinism; strictness of config errors; `check-data` exit codes |

Statistical tests use fixed seeds and tolerances sized from the sampling error at the sample sizes
used, so they are deterministic rather than flaky.
