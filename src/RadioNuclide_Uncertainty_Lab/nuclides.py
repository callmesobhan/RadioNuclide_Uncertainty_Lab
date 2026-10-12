"""Radionuclide reference data (only entries backed by a cited source).

Every entry carries its provenance; add a nuclide only with a source you can cite. Y-90 and
I-131 are deliberately absent until their values are taken from a verified evaluation
(see ``docs/data-sources.md``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Nuclide:
    symbol: str
    half_life_d: float
    half_life_unc_d: float  # standard (1-sigma) uncertainty
    delta_e_mev: float  # mean non-penetrating (electron) energy emitted per decay
    delta_ph_mev: float  # mean photon energy emitted per decay
    delta_e_alt_mev: float | None  # alternative evaluation, to indicate data uncertainty
    source: str

    @property
    def half_life_h(self) -> float:
        return self.half_life_d * 24.0


NUCLIDES: dict[str, Nuclide] = {
    "Lu-177": Nuclide(
        symbol="Lu-177",
        half_life_d=6.6443,
        half_life_unc_d=0.0009,
        delta_e_mev=0.1479,
        delta_ph_mev=0.0351,
        delta_e_alt_mev=0.1465,
        source=(
            "Half-life: LNHB DDEP evaluation (Kellett, Mougeot; revised 26/11/2025), 6.6443(9) d, "
            "https://www.lnhb.fr/nuclides/Lu-177_tables.pdf; cite Mougeot et al., Metrologia 63 "
            "(2026) 019001. Staanum 2021 uses 6.647 d, an older evaluation (0.04 % higher). "
            "Energies per decay: ICRP Publication 107 as tabulated in Sjogreen Gleisner 2022 "
            "(EANM), Table 1 (electrons 147.9 keV, photons 35.1 keV), checked against bounds "
            "computed from the DDEP emission tables; alternative electron value 0.1465 MeV from "
            "NNDC as quoted by Staanum 2021."
        ),
    ),
}


def get_nuclide(symbol: str) -> Nuclide:
    try:
        return NUCLIDES[symbol]
    except KeyError:
        raise KeyError(f"unknown nuclide {symbol!r}; known: {sorted(NUCLIDES)}") from None
