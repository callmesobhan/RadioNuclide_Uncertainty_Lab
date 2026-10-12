"""Reference datasets shipped with the package, with provenance and load-time validation.

A dataset is a JSON file in ``data/`` produced by a tool in ``tools/`` (so it can be regenerated
from the original source) and carries its provenance: archive, member file, CRC-32, SHA-256,
database version, citation and terms. Loading checks the *structure* and the *physics that cannot
be wrong* (finite, non-negative, right units, masses positive). Softer consistency checks across
sources (reciprocity, energy conservation, agreement with a second database) live in
:mod:`RadioNuclide_Uncertainty_Lab.datacheck`, where a failure excludes the data from use.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

EXPECTED_UNITS = {"s_value": "Gy/(MBq h)", "mass": "kg"}
_REQUIRED = {
    "dataset",
    "nuclide",
    "phantom",
    "units",
    "regions",
    "mass_kg",
    "s_total",
    "s_photon",
    "provenance",
}
_PROVENANCE_REQUIRED = {"archive_member", "member_crc32", "member_sha256", "citation", "terms"}


@dataclass(frozen=True)
class SValueSet:
    """Organ S-values S(target <- source) in Gy/(MBq h) for one nuclide and phantom."""

    dataset: str
    nuclide: str
    phantom: str
    regions: tuple[str, ...]
    mass_kg: Mapping[str, float]
    s_total: Mapping[str, Mapping[str, float]]  # [target][source]
    s_photon: Mapping[str, Mapping[str, float]]
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def s(self, target: str, source: str) -> float:
        return self.s_total[target][source]

    def self_s(self, region: str) -> float:
        return self.s_total[region][region]

    def mass(self, region: str) -> float:
        return self.mass_kg[region]

    def cross_table(self, regions: Mapping[str, str]) -> dict[str, dict[str, float]]:
        """Cross-organ S-values between model organs (self terms excluded).

        ``regions`` maps model organ -> dataset region; returns ``{target: {source: S}}``."""
        return {
            t: {s: self.s_total[rt][rs] for s, rs in regions.items() if s != t}
            for t, rt in regions.items()
        }

    @property
    def fingerprint(self) -> str:
        """Short identifier of the exact source file this data came from."""
        return f"{self.provenance.get('member_sha256', '?')[:12]}"


def load_svalues(name: str) -> SValueSet:
    """Load and structurally validate a packaged S-value dataset by name."""
    path = resources.files(__package__) / "data" / f"{name}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KeyError(
            f"unknown S-value dataset {name!r}; available: {available_svalue_sets()}"
        ) from None
    return _validate_and_build(raw, name)


def available_svalue_sets() -> list[str]:
    folder = resources.files(__package__) / "data"
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


@cache
def cached_svalues(name: str) -> SValueSet:
    return load_svalues(name)


def _validate_and_build(raw: Mapping[str, Any], name: str) -> SValueSet:
    missing = _REQUIRED - set(raw)
    if missing:
        raise ValueError(f"dataset {name!r}: missing keys {sorted(missing)}")
    if raw["dataset"] != name:
        raise ValueError(f"dataset {name!r}: file declares itself as {raw['dataset']!r}")
    if raw["units"] != EXPECTED_UNITS:
        raise ValueError(f"dataset {name!r}: units {raw['units']} != {EXPECTED_UNITS}")
    missing_prov = _PROVENANCE_REQUIRED - set(raw["provenance"])
    if missing_prov:
        raise ValueError(f"dataset {name!r}: provenance lacks {sorted(missing_prov)}")
    regions = tuple(raw["regions"])
    if len(set(regions)) != len(regions) or not regions:
        raise ValueError(f"dataset {name!r}: regions must be unique and non-empty")
    for r in regions:
        m = raw["mass_kg"].get(r)
        if m is None or not math.isfinite(m) or m <= 0:
            raise ValueError(
                f"dataset {name!r}: mass of {r!r} must be positive and finite, got {m!r}"
            )
    for table in ("s_total", "s_photon"):
        if set(raw[table]) != set(regions):
            raise ValueError(f"dataset {name!r}: {table} targets differ from regions")
        for t in regions:
            if set(raw[table][t]) != set(regions):
                raise ValueError(f"dataset {name!r}: {table}[{t!r}] sources differ from regions")
            for s, v in raw[table][t].items():
                if not (isinstance(v, int | float) and math.isfinite(v) and v >= 0):
                    raise ValueError(
                        f"dataset {name!r}: {table}[{t!r}][{s!r}] = {v!r} is not a "
                        "finite non-negative number"
                    )
    for t in regions:
        for s in regions:
            if raw["s_photon"][t][s] > raw["s_total"][t][s] * (1 + 1e-9):
                raise ValueError(f"dataset {name!r}: photon S exceeds total S for {t!r} <- {s!r}")
    return SValueSet(
        dataset=name,
        nuclide=raw["nuclide"],
        phantom=raw["phantom"],
        regions=regions,
        mass_kg=dict(raw["mass_kg"]),
        s_total={t: dict(row) for t, row in raw["s_total"].items()},
        s_photon={t: dict(row) for t, row in raw["s_photon"].items()},
        provenance=dict(raw["provenance"]),
    )
