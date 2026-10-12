# Data sources

Every number in the shipped scenario ([`examples/configs/lu177_dotatate_multi_organ.yaml`](../examples/configs/lu177_dotatate_multi_organ.yaml)),
in [`nuclides.py`](../src/RadioNuclide_Uncertainty_Lab/nuclides.py) and in the S-value dataset comes from one of the
publications or databases below, is *derived* from them (§3, enforced by tests), or is an
*explicitly stated assumption* (§4). There are no unsourced placeholders. Nuclides for which no
verified source was available (Y-90, I-131) were removed from the registry rather than left as guesses.
Values in the nuclide registry are tied to their claims by a check, so they cannot go stale.

Machine-readable form: [`claims.yaml`](../src/RadioNuclide_Uncertainty_Lab/data/claims.yaml) holds every
published number with its location in the source, plus the checks that guard it (§6).

## 1. Sources

| Key | Citation | What is used |
|---|---|---|
| **[Staanum2021]** | Staanum et al. *Practical kidney dosimetry in peptide receptor radionuclide therapy using [177Lu]Lu-DOTATOC and [177Lu]Lu-DOTATATE with focus on uncertainty estimates.* EJNMMI Physics 2021;8:78. PMC8590641. | Table 2 uncertainty budget (every term in the config); kidney mean specific absorbed dose 0.46 Gy/GBq and mean effective half-life 2.27 d for [177Lu]Lu-DOTATATE (n = 56 fractions); 7400 MBq per fraction; Fig. 6b fit example (relative uncertainty C0 8.9 %, λ 8.7 %, AUC 6.1 %); physical half-life 6.647 d; electron energy per decay 0.1479 MeV (NNDC 0.1465 MeV). |
| **[Gear2018]** | Gear JI, Cox MG, Gustafsson J, Sjögreen Gleisner K, Murray I, Glatting G, Konijnenberg M, Flux GD. *EANM practical guidance on uncertainty analysis for molecular radiotherapy absorbed dose calculations.* Eur J Nucl Med Mol Imaging 2018;45:2456–2474. doi:10.1007/s00259-018-4136-7. PMC6208822. | Worked example (90Y-DOTATATE, two lesions): Table 6 activities, Tables 7–8 fit and covariance, Tables 10–13 S-factor model S = 0.429·m^−0.961 Gy/(MBq h), Ã, S, their correlation, D and u(D). Used for benchmarks and `examples/02_fit_to_dose.py`, not for the scenario's values. |
| **[Gleisner2022]** | Sjögreen Gleisner K, Chouin N, Gabina PM, et al. *EANM dosimetry committee recommendations for dosimetry of 177Lu-labelled somatostatin-receptor- and PSMA-targeting ligands.* Eur J Nucl Med Mol Imaging 2022;49:1778–1809. doi:10.1007/s00259-022-05727-7. | Table 2: self-dose S-values (IDAC-Dose 2.1, adult male phantom) and masses, used as an **independent cross-check** of the MIRDcalc data; Table 1: ICRP 107 energies per decay (electrons 147.9 keV, photons 35.1 keV); text: rest-of-body contribution ≈ 2 % of kidney dose. |
| **[MIRDcalc]** | Kesner AL, Carter LM, Ocampo Ramos JC, et al. *MIRD Pamphlet No. 28, Part 1: MIRDcalc.* J Nucl Med 2023;64(10):1524–1531. doi:10.2967/jnumed.122.264225 (the citation the database page asks for). S-value database, https://mirdsoft.org/s-value-database, archive `MIRDcalcDB-v20251023-1224x12-csv.zip`, member `SDATA_Lu-177_ICRP Adult Male.csv`, database version 20260206. | The **S-value matrices** (gamma + beta + alpha) and target masses: `lu177_adult_male` (kidneys, liver, spleen, red marrow; used by the scenario) and `lu177_adult_female` (eight regions; used to validate the cross terms), ICRP 110 phantoms, ICRP 107 decay data. See §2 and §4. |
| **[Santoro2018]** | Santoro L, Mora-Ramirez E, Trauchessec D, Chouaf S, Eustache P, Pouget J-P, Kotzki P-O, Bardiès M, Deshayes E. *Implementation of patient dosimetry in the clinical practice after targeted radiotherapy using [177Lu-[DOTA0, Tyr3]-octreotate.* EJNMMI Research 2018;8:103. doi:10.1186/s13550-018-0459-4 (CC BY 4.0; title and authors as given by Crossref). | Mean organ doses over all patients and cycles (mGy/MBq): kidneys 0.43, liver 0.54 (SD 0.58), spleen 0.61 (12 patients, SPECT/CT at 4, 24, 72 and 192 h). |
| **[Bodei2025]** | Bodei L, Cremonesi M, Ferrari M, et al. *Dosimetry of [177Lu]Lu-DOTATATE in patients with advanced midgut neuroendocrine tumors: results from a substudy of the phase III NETTER-1 trial.* J Nucl Med 2025;66(3):449–456. doi:10.2967/jnumed.124.268903 (CC BY 4.0). | Cross-check only: Gy/GBq mean kidneys 0.65, liver 0.30, spleen 0.85; mean kidney TIA coefficient 2.7 h. |
| **[DDEP]** | LNHB DDEP evaluation of 177Lu (M.A. Kellett, X. Mougeot; 07/03/2003 – 26/11/2025), `Lu-177_tables.pdf`, https://www.lnhb.fr/nuclides/Lu-177_tables.pdf (file 834,937 bytes, SHA-256 `3f4f12c406692dbe…`, retrieved 2026-10-12). Please cite X. Mougeot et al., Metrologia 63 (2026) 019001, doi:10.1088/1681-7575/ae3733. | **Primary** nuclear data: T½ = **6.6443 (9) d**; β⁻ transition probabilities and average energies; conversion- and Auger-electron emissions (used to bound the mean electron energy per decay). Replaces the 6.647(4) d first taken from a secondary source, which was an older evaluation. |
| **[OpenDose2020]** | Chauvin M et al. *OpenDose: open-access resource for nuclear medicine dosimetry.* J Nucl Med 2020;61(10):1514. doi:10.2967/jnumed.119.240366. Table 1. | Lu-177 S-values for a **liver source in the ICRP 110 female phantom**, as computed by OpenDose and by IDAC-Dose 2.1, for six targets: the **independent second source for the cross-organ terms** (§2). OLINDA/EXM 2.0 values in the same table use a different phantom and are not used. |

**How the numbers were read.** The papers' full texts were retrieved through Europe PMC's open-access
full-text service and key values extracted with automated assistance; the MIRDcalc numbers were
parsed from the CSV by a script (below). Because transcription is the weak point, every extracted
value is a claim guarded by checks (§6); please still spot-check values you intend to cite
against the PDF.

## 2. How the S-value datasets were obtained, and how the cross terms are validated

The EANM Lu-177 recommendations tabulate only **self** S-values. Cross-organ S-values were obtained
from the MIRDcalc S-value database, which publishes 14,689 CSV files (1,224 nuclides × 12
phantoms) in a 548 MB archive. Only two files were needed, so
[`tools/extract_mirdcalc_svalues.py`](../tools/extract_mirdcalc_svalues.py) read the archive's directory and
those members by HTTP range requests (2.1 MB per file transferred), verified their CRC-32, parsed them
**by label, not by fixed position**, and wrote
[`lu177_adult_male.json`](../src/RadioNuclide_Uncertainty_Lab/data/lu177_adult_male.json) (4 regions, CRC-32
`4302f61d`) and [`lu177_adult_female.json`](../src/RadioNuclide_Uncertainty_Lab/data/lu177_adult_female.json)
(8 regions, CRC-32 `8561c4f6`): the S-value matrix (total and photon part), the masses, and the
provenance (archive URL, size and modification time, member name, CRC-32, SHA-256, database version, citation,
terms). **The downloaded files were deleted afterwards**; the tool regenerates the datasets from the source
at any time.

Conventions verified on the file: rows are source regions and columns targets; units mGy/(MBq s),
converted by the exact factor 3.6 to Gy/(MBq h); the mass used for a self-dose S-value is the target
"tissue & blood" mass (kidneys 422 g, liver 2360 g, spleen 228.4 g, red marrow 1394 g), which is
exactly what IDAC-Dose uses in [Gleisner2022] Table 2.

**What validates the numbers.** The scenario's phantom is the *male* one, and no independent source tabulates
male Lu-177 cross terms that I could reach. Three independent lines of evidence stand in for it:

1. **Same pipeline, independent codes (female phantom).** [OpenDose2020] Table 1 gives Lu-177 S-values for a
   liver source in the ICRP 110 female phantom, from OpenDose and from IDAC-Dose 2.1. For six targets spanning
   more than four orders of magnitude (8.4 × 10⁻¹⁰ to 1.4 × 10⁻⁵ mGy/(MBq s)), MIRDcalc agrees with **IDAC-Dose within
   0.40 %** and with **OpenDose within 0.52 %** (liver→spleen, the cross term of the scenario, within 0.09 %
   and 0.37 %). The table prints no units; the agreement of all six values fixes mGy/(MBq s).
   The one disagreement, OpenDose's liver *self* term (29 % above MIRDcalc and IDAC-Dose; MIRDcalc is 22.5 % below it), is explained by
   the paper itself (mass adjustment in the ICRP 133 SAFs); it is declared a known conflict and the OpenDose
   claim is excluded (§6).
2. **Physics on both datasets.** Reciprocity, S(t←s) = S(s←t), holds to 0.20 % for the scenario's three
   organs (and ≤ 4.7 % for every pair above the Monte Carlo noise floor); absorbed energy never exceeds
   emitted energy (worst 0.85).
3. **The quantity that matters, against the literature.** The cross-organ S-values put **1.80 %** of the
   anchored kidney dose on the liver and spleen; [Gleisner2022] states "about 2 %".

What is *not* independently validated: the male cross terms themselves are checked by (2) and (3) only, and
the OpenDose and IDAC cross terms are for the female phantom. A second male source would close that gap
(roadmap).

Which other sources were tried and why they were not used: the OpenDose website refused connections from the
sandbox used (so its *paper* is the source above); the RADAR site presented a TLS certificate for a different
host (not bypassed) and uses older phantoms; the MIRDcalc *software* needs a registered account (not created
here) and IDAC-Dose is a desktop application without browsable tables; the IAEA nuclear-data service sits
behind a bot challenge (not bypassed); the ICRP annexes found contain intake dose coefficients, not
nuclear-medicine S-values; published patient-level cross-dose tables were incomplete or inconsistent.

## 3. Derived nominal values

The scenario is anchored so that it reproduces published cohort mean doses at nominal inputs.
The organ TIA coefficients `T` (h) solve the whole linear system

    S · T = D          S[t, s] = S(t ← s) from the dataset (self and cross), D = published dose per activity

(`mird.tia_coefficients_from_doses`), so that *total* dose (self + cross) equals the published value.
Solving the system instead of dividing each dose by its self S-value matters at the 2 % level for
the kidneys.

| Organ | Published mean D/A | Result T (h) | Kinetics inputs |
|---|---|---|---|
| kidneys | 0.46 Gy/GBq [Staanum2021] | 2.2061 | uptake fraction `f = T·ln2 / T_eff = 0.028068` with T_eff = 2.27 d = 54.48 h [Staanum2021] |
| liver | 0.54 Gy/GBq [Santoro2018] | 14.287 | direct TIA coefficient |
| spleen | 0.61 Gy/GBq [Santoro2018] | 1.6054 | direct TIA coefficient |

`tests/test_data_integrity.py` recomputes these from the dataset and the claims and fails if the
config differs by more than the rounding of the printed digits, so they cannot drift.

Independent corroboration: the derived kidney coefficient (2.21 h) sits between Santoro's implied value
(0.43/0.205 = 2.1 h) and Bodei's reported mean (2.7 ± 1.5 h); the cross-organ S-values give **1.8 %**
of the kidney dose from the liver and spleen, matching "about 2 %" in [Gleisner2022].

Kidney fit: relative SDs 8.9 % (C0) and 8.7 % (λ) with a reported AUC uncertainty of 6.1 % imply
`r(C0, λ) = (8.9² + 8.7² − 6.1²) / (2·8.9·8.7) = +0.76`. Because T_eff = ln2/λ the correlation of
`(uptake_fraction, effective_half_life_h)` is **−0.76** (claim `staanum2021.derived.rho_c0_teff`,
checked against the three inputs). Marginals are lognormal with GSD = exp(√ln(1 + cv²)) = 1.0929
and 1.0907.

## 4. Terms of the S-value source

The MIRDsoft site states that its content is copyright protected and may be used for "your own
personal and noncommercial use" without prior written consent, provides the data "as is", and names
no open-data licence for the S-value database; it asks users to cite the MIRDcalc pamphlet above.

What this repository stores from it: two small organ S-value matrices (4 and 8 regions; total and photon
part), their masses and the provenance, as **numerical results with citation**, plus a tool to regenerate
them. Whether that is acceptable for a public repository under the Apache-2.0 licence is a question for the
project owner (and, if in doubt, for MIRDsoft); `NOTICE` states that these files keep their source's terms.

To remove them: delete `data/lu177_adult_male.json` and `data/lu177_adult_female.json`; delete the
`kind: refdata` checks that name those datasets (`svalues.*`, and `svalues.male.kidney_cross_share_vs_eanm`) from
`claims.yaml`, since `check-data` fails on a check whose dataset is missing; and in a config replace `s_values:`
and the organs' `region:` by explicit `s_self_gy_per_mbq_h` / `mass_ref_kg` values (the model then runs
without cross-organ dose). Or keep the files out of version control and regenerate them locally with
`tools/extract_mirdcalc_svalues.py` from your own copy of the source.

## 5. Assumptions (stated, not sourced)

1. **The kidney budget is applied to liver and spleen.** [Staanum2021] analysed kidneys. The
   partial-volume term (7 %) and delineation term (3 %) are likely smaller for large organs, so the
   liver/spleen budgets here are conservative. Replace them if you have organ-specific values.
2. **Shared vs organ-specific split.** Dose-calibrator and SPECT-system terms are treated as one error
   common to all organs measured on the same system; delineation, partial-volume, time-activity-model,
   fit and dose-factor terms are independent per organ. This affects organ correlations and ratio
   uncertainties, **not** any single organ's marginal uncertainty.
3. **Masses fixed at the phantom reference values.** [Staanum2021] works in activity concentration,
   where volume largely cancels; its 3 % delineation term already represents the net effect on dose.
   Adding an independent mass uncertainty on top would double count.
4. **Phantom, not patient.** The S-values are for the ICRP 110 reference adult male; the published
   cohorts include women and different organ sizes. Mass scaling (`mass_kg`) handles size for self-dose
   only; cross-dose terms are used unscaled (they are ~2 % of the dose, and validated on the female phantom, §2).
5. **Cohort means are used as the nominal dose.** Spread between cohorts (kidneys 0.43–0.65 Gy/GBq
   across [Santoro2018], [Staanum2021], [Bodei2025]) is inter-patient/inter-cohort variability and is
   not in the budget.
6. **Not modelled:** remainder of body, tumours, red marrow kinetics.

## 6. Known discrepancies in the sources, and how they are handled

Found while extracting, recorded as claims, and kept failing on purpose (`expect: conflict`): the claims
are **excluded**, nothing in the scenario uses them, and CI turns red if a source is ever corrected
(`UNEXPECTED_PASS`), so a human reviews the resolution.

| Check | Source says | Conflict |
|---|---|---|
| `staanum.abstract_vs_results.dotatoc_mean` / `_median` | Abstract: DOTATOC kidney mean 0.37, median 0.38 Gy/GBq | Results: median 0.37, mean 0.38 (swapped) |
| `staanum.one_spect_lambda_uncertainty` | Text: 1-SPECT λ uncertainty 5.5 % | Table 2 footnote d: 9 % |
| `gear.s_model_c2_uncertainty` | Table 10 lists 1.2 % for c2 | 0.0051/0.961 = 0.53 % |
| `gear.s_model_c1_uncertainty` | Table 10 lists 0.4 % for c1 | 0.0037/0.429 = 0.86 % |
| `svalues.female.liver_self_vs_opendose` | OpenDose: Liver←Liver 1.753 × 10⁻⁵ mGy/(MBq s) | MIRDcalc and IDAC-Dose 1.36 × 10⁻⁵ (−22.5 %); the paper attributes it to the ICRP 133 mass adjustment. Self term only; the cross terms agree to 0.5 % |

Also noted, not conflicts: Staanum's total of 13 % is the quadrature of its terms (12.1 %) with each term
rounded up; Gear's Table 9 and 12 print 5,933 and 5,932 MBq h for the same quantity (rounding); Staanum
uses T½ = 6.647 d, an older DDEP evaluation (0.04 % above the current 6.6443(9) d), within the older
evaluation's uncertainty. This last one was found by replacing a secondary-source half-life with the primary
evaluation, and a registry check (`nuclides.registry_matches_claims`) now fails if `nuclides.py` ever differs
from the claims again.

## 7. Consistency checks that must pass

`radionuclide-uncertainty-lab check-data` (and the test suite) evaluates 40 checks over 131 claims: 34 are
expected to pass, 6 are declared known conflicts (§6). The passing ones include:

- internal consistency of a source: Staanum's terms add to the stated total; Gear's S-factor, nominal dose,
  first-order uncertainty and fit-covariance arithmetic;
- agreement between sources: IDAC-Dose self S-values and masses vs MIRDcalc (≤ 0.4 %); **MIRDcalc cross terms vs
  IDAC-Dose (≤ 0.40 %) and OpenDose (≤ 0.52 %)**; two sources for the electron energy; kidney dose of two cohorts
  within 10 %;
- **primary nuclear data**: T½ from the current DDEP evaluation; the ICRP 107 mean electron energy per decay
  (147.9 keV) and the NNDC value (146.5 keV) both lie inside the interval **[140.7, 151.0] keV** built from the
  DDEP β⁻ averages and the minimum/maximum energy of each conversion- and Auger-electron group;
- physics on the datasets: **reciprocity** (noise-aware: only cross terms ≥ 10⁻³ of the self terms are tested;
  asymmetry grows to 13 % for terms near 10⁻⁵ of the self terms, which is Monte Carlo noise and below 0.1 % of any
  dose; the scenario's three organs are tested separately at 1 %) and **energy conservation** (≤ 85 % of the
  emitted energy appears in the four listed targets);
- the scenario's cross-organ share of the kidney dose (1.80 %) against the EANM's "about 2 %" (±0.7 pp);
- the nuclide registry in code against its claims.

## 8. Rules for adding data

1. Every uncertain input gets a `source:` string **and** a `claim:` id; add the claim, with its
   location, to `claims.yaml` first.
2. Prefer the quoted *standard* uncertainty over a range; if only a range is known use
   `uniform`/`triangular` and say so. Convert expanded (k = 2) uncertainties to standard.
3. Add at least one check that ties the new claim to something else (another source, arithmetic, physics).
   A claim nothing can contradict is an unverified claim.
4. A new reference dataset needs a generator in `tools/`, provenance, and checks registered with its
   `dataset:` name, otherwise the model refuses it.
5. Nuclide entries need a half-life uncertainty and a citation.
6. Derived values go in this file with their derivation and a test that recomputes them.
7. Never tune a distribution to make a budget look better.
