"""YAML configuration -> ``UncertainModel`` + run settings.

The config is the unit of reproducibility: it lists every uncertain input with its
distribution and *source*, every constant, and the run settings (sample sizes, seed).
Its SHA-256 goes into every report.

Per-organ templating
--------------------
A parameter key containing ``{organ}`` is expanded once per organ (or per organ in its
``organs:`` list), substituting the organ name in the key and in every string field, e.g.::

    "{organ}.activity_scale.delineation": {dist: normal, mean: 1, rel_sd: 0.03,
                                          group: "delineation.{organ}", ...}

Correlation entries may use ``{organ}`` the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .datacheck import verify_correlation_claim, verify_parameter_claim
from .distributions import from_spec
from .inputs import Parameter
from .model import UncertainModel
from .models import MODEL_BUILDERS
from .sampling import METHODS

_COMMON_KEYS = {"name", "model", "fixed", "parameters", "correlations", "run"}
_PARAM_META = {"group", "unit", "source", "description", "claim"}
_RUN_KEYS = {"n", "method", "seed", "sobol_n", "level"}


@dataclass(frozen=True)
class RunSettings:
    n: int = 2**14
    method: str = "sobol"
    seed: int | None = None
    sobol_n: int = 2**13
    level: float = 0.95


@dataclass(frozen=True)
class LoadedConfig:
    model: UncertainModel
    run: RunSettings
    sha256: str
    raw: Mapping[str, Any]
    claims_used: tuple[str, ...] = ()

    def data_integrity(self) -> dict[str, Any]:
        """What published data this config relies on, for the report's provenance."""
        from .datacheck import get_registry

        reg = get_registry()
        info = self.model.metadata.get("s_values")
        return {
            "claims_used": list(self.claims_used),
            "checks_passed": sum(r.status == "PASS" for r in reg.results),
            "checks_known_conflict": sum(r.status == "KNOWN_CONFLICT" for r in reg.results),
            "excluded_claims": sorted(reg.excluded),
            "s_value_dataset": info,
        }


def load_config(source: str | Path | Mapping[str, Any]) -> LoadedConfig:
    if isinstance(source, Mapping):
        raw = dict(source)
        text = yaml.safe_dump(raw, sort_keys=True)
    else:
        text = Path(source).read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping at the top level")
    for key in ("model", "parameters"):
        if key not in raw:
            raise ValueError(f"config is missing required key {key!r}")
    if raw["model"] not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {raw['model']!r}; choose from {sorted(MODEL_BUILDERS)}")
    spec = MODEL_BUILDERS[raw["model"]]
    extra = set(raw) - _COMMON_KEYS - spec.options
    if extra:
        raise ValueError(f"unknown top-level keys: {sorted(extra)}")

    organ_names = list((raw.get("organs") or {}).keys())
    parameters = [
        _parameter(name, s) for name, s in _expand_templates(raw["parameters"], organ_names).items()
    ]
    correlations = {}
    claims_used = {p.claim for p in parameters if p.claim}
    for entry in _expand_correlations(raw.get("correlations", []), organ_names):
        a, b = entry["between"]
        correlations[(a, b)] = float(entry["rho"])
        if entry.get("claim"):
            verify_correlation_claim((a, b), correlations[(a, b)], entry["claim"])
            claims_used.add(entry["claim"])

    model = spec.builder(
        parameters=parameters,
        correlations=correlations,
        fixed=raw.get("fixed", {}),
        name=raw.get("name"),
        **{k: raw[k] for k in spec.options if k in raw},
    )
    run_raw = dict(raw.get("run", {}))
    bad = set(run_raw) - _RUN_KEYS
    if bad:
        raise ValueError(f"unknown run settings: {sorted(bad)}")
    run = RunSettings(**run_raw)
    if run.method not in METHODS:
        raise ValueError(f"run.method must be one of {METHODS}")
    return LoadedConfig(
        model, run, hashlib.sha256(text.encode()).hexdigest(), raw, tuple(sorted(claims_used))
    )


def _substitute(value: Any, organ: str) -> Any:
    return value.replace("{organ}", organ) if isinstance(value, str) else value


def _targets(spec: dict[str, Any], organ_names: list[str], what: str) -> list[str]:
    targets = spec.pop("organs", organ_names)
    unknown = [o for o in targets if o not in organ_names]
    if unknown:
        raise ValueError(f"{what}: unknown organs {unknown}")
    if not organ_names:
        raise ValueError(f"{what}: '{{organ}}' used but the config defines no organs")
    return list(targets)


def _expand_templates(
    params: Mapping[str, Mapping[str, Any]], organ_names: list[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, spec in params.items():
        spec = dict(spec)
        if "{organ}" in key:
            for organ in _targets(spec, organ_names, f"parameter {key!r}"):
                expanded = {k: _substitute(v, organ) for k, v in spec.items()}
                out[key.replace("{organ}", organ)] = expanded
        else:
            if "organs" in spec:
                raise ValueError(f"parameter {key!r}: 'organs' is only valid with '{{organ}}' keys")
            if key in out:
                raise ValueError(f"duplicate parameter {key!r} after expansion")
            out[key] = spec
    return out


def _expand_correlations(
    entries: list[Mapping[str, Any]], organ_names: list[str]
) -> list[dict[str, Any]]:
    out = []
    for entry in entries:
        entry = dict(entry)
        if any("{organ}" in s for s in entry["between"]):
            for organ in _targets(entry, organ_names, "correlation"):
                out.append(
                    {
                        "between": [_substitute(s, organ) for s in entry["between"]],
                        "rho": entry["rho"],
                    }
                )
        else:
            out.append(entry)
    return out


def _parameter(name: str, spec: Mapping[str, Any]) -> Parameter:
    spec = dict(spec)
    meta = {k: spec.pop(k) for k in list(spec) if k in _PARAM_META}
    try:
        dist = from_spec(spec)
    except ValueError as err:
        raise ValueError(f"parameter {name!r}: {err}") from None
    if meta.get("claim"):
        verify_parameter_claim(name, dist, meta["claim"])
    return Parameter(name=name, dist=dist, **meta)
