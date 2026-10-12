"""Ready-made dose models, selectable by name from a config file.

Writing your own is just ``UncertainModel(name, InputSpace(...), func, outputs=...)``; see
``examples/02_fit_to_dose.py``. Builders registered here share one calling convention so the
config loader can call them uniformly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .inputs import InputSpace, Parameter
from .kinetics import LN2
from .model import UncertainModel
from .refdata import SValueSet, cached_svalues

# --------------------------------------------------------------------------------------------
# Multi-organ MIRD model
# --------------------------------------------------------------------------------------------

#: quantities each kinetics type needs, per organ (names are ``<organ>.<quantity>``)
KINETICS: dict[str, tuple[str, ...]] = {
    "mono_exponential": ("uptake_fraction", "effective_half_life_h"),
    "tia_coefficient": ("tia_coefficient_h",),
}
_SCALE_KINDS = ("activity_scale", "dose_scale")
_ORGAN_KEYS = {
    "s_self_gy_per_mbq_h",
    "mass_ref_kg",
    "kinetics",
    "mass_exponent",
    "source",
    "region",
}


@dataclass(frozen=True)
class Organ:
    """A source region that is also a target.

    ``s_self_gy_per_mbq_h`` is S(organ <- organ) for a reference organ of mass ``mass_ref_kg``;
    for a patient organ of mass ``m`` it is scaled as ``S * (m / mass_ref)^(-mass_exponent)``.
    Exponent 1 is exact for electron self-dose in a large organ (absorbed energy is the same,
    mass differs); small lesions follow other power laws (see Gear et al. 2018, S = c1 m^-c2).

    With ``region`` set, the self S-value and the reference mass are taken from the packaged
    S-value dataset (e.g. ``Kidneys``) instead of being typed in, and cross-organ S-values to other
    organs with a ``region`` come from the same dataset, so all of them share one phantom.
    """

    name: str
    s_self_gy_per_mbq_h: float
    mass_ref_kg: float
    kinetics: str
    mass_exponent: float = 1.0
    source: str = field(default="", compare=False)
    region: str | None = field(default=None, compare=False)


def organs_from_spec(
    spec: Mapping[str, Mapping[str, object]], dataset: SValueSet | None = None
) -> list[Organ]:
    organs = []
    for name, d in spec.items():
        if not name or "." in name or name in _SCALE_KINDS:
            raise ValueError(f"invalid organ name {name!r} (no dots; not a reserved word)")
        extra = set(d) - _ORGAN_KEYS
        if extra:
            raise ValueError(f"organ {name!r}: unknown keys {sorted(extra)}")
        d = dict(d)
        if "region" in d:
            if dataset is None:
                raise ValueError(f"organ {name!r}: 'region' needs the top-level 's_values' dataset")
            if "s_self_gy_per_mbq_h" in d or "mass_ref_kg" in d:
                raise ValueError(
                    f"organ {name!r}: give either 'region' or explicit "
                    "s_self_gy_per_mbq_h/mass_ref_kg"
                )
            if d["region"] not in dataset.regions:
                raise ValueError(
                    f"organ {name!r}: region {d['region']!r} not in dataset {dataset.dataset!r}; "
                    f"available: {list(dataset.regions)}"
                )
            d["s_self_gy_per_mbq_h"] = dataset.self_s(d["region"])
            d["mass_ref_kg"] = dataset.mass(d["region"])
        try:
            organ = Organ(name=name, **d)  # type: ignore[arg-type]
        except TypeError as err:
            raise ValueError(f"organ {name!r}: {err}") from None
        if organ.kinetics not in KINETICS:
            raise ValueError(f"organ {name!r}: kinetics must be one of {sorted(KINETICS)}")
        if organ.s_self_gy_per_mbq_h <= 0 or organ.mass_ref_kg <= 0 or organ.mass_exponent <= 0:
            raise ValueError(f"organ {name!r}: S-value, reference mass and exponent must be > 0")
        organs.append(organ)
    if not organs:
        raise ValueError("at least one organ is required")
    return organs


def _classify(name: str, organ_names: Sequence[str]) -> tuple[str, str | None] | None:
    """``('activity_scale', None)`` for global factors, ``('dose_scale', 'liver')`` for
    organ-specific ones, ``None`` if ``name`` is not a scale factor."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in organ_names and parts[1] in _SCALE_KINDS:
        return parts[1], parts[0]
    if len(parts) >= 2 and parts[0] in _SCALE_KINDS:
        return parts[0], None
    return None


def multi_organ(
    parameters: Sequence[Parameter],
    correlations: Mapping[tuple[str, str], float] | None = None,
    fixed: Mapping[str, float] | None = None,
    name: str | None = None,
    *,
    organs: Mapping[str, Mapping[str, object]] | Sequence[Organ],
    cross_s_gy_per_mbq_h: Mapping[str, Mapping[str, float]] | None = None,
    ratios: Sequence[Mapping[str, str]] | None = None,
    nuclide: str | None = None,
    s_values: str | None = None,
    cross_dose: bool = True,
) -> UncertainModel:
    """MIRD organ-level dose for several source/target organs, one output per organ.

    ``D_t = scale_t * ( S(t<-t) * TIA_t + sum_{s != t} S(t<-s) * TIA_s )`` with the time-integrated
    activity of source ``s`` given by one of two kinetics types:

    * ``mono_exponential``: ``TIA = f * A_admin * T_eff / ln2`` (``uptake_fraction`` f at t = 0 and
      ``effective_half_life_h``; physical decay is part of the measured effective half-life);
    * ``tia_coefficient``: ``TIA = c * A_admin`` (``tia_coefficient_h``, hours).

    Quantities are named ``<organ>.<quantity>`` and must each be given exactly once, either as an
    uncertain ``Parameter`` or as a constant in ``fixed``. ``<organ>.mass_kg`` is optional and
    defaults to the organ's reference mass. ``administered_activity_mbq`` is always a constant.

    Multiplicative error factors (nominal 1) are picked up by name and can be shared or specific:

    * ``activity_scale.<label>`` multiplies **every** organ's TIA (a shared systematic error, e.g.
      scanner calibration); ``<organ>.activity_scale.<label>`` only that organ's;
    * ``dose_scale.<label>`` / ``<organ>.dose_scale.<label>`` multiply the dose in the same way.

    Shared factors cancel in dose *ratios*; that is why ``ratios`` outputs are offered:
    ``[{name: kidneys_to_spleen, numerator: kidneys, denominator: spleen}]``.

    ``cross_s_gy_per_mbq_h`` is an optional ``{target: {source: S}}`` table of cross-organ
    S-values in Gy/(MBq h) (constants; their uncertainty can be expressed with ``dose_scale``).
    ``s_values`` names a packaged dataset (see ``refdata``): organs with a ``region`` take their
    self S-value and mass from it, and with ``cross_dose`` (default) the cross-organ terms between
    such organs are added from the same dataset. ``cross_dose=False`` reproduces a
    self-dose-only model.
    """
    if s_values:
        from .datacheck import require_trusted_dataset

        require_trusted_dataset(s_values)
    dataset = cached_svalues(s_values) if s_values else None
    organ_list = (
        list(organs) if not isinstance(organs, Mapping) else organs_from_spec(organs, dataset)  # type: ignore[arg-type]
    )
    names = [o.name for o in organ_list]
    by_name = {o.name: o for o in organ_list}
    fixed = {k: float(v) for k, v in (fixed or {}).items()}
    uncertain = {p.name for p in parameters}

    if "administered_activity_mbq" not in fixed:
        raise ValueError("fixed must contain 'administered_activity_mbq'")
    both = uncertain & set(fixed)
    if both:
        raise ValueError(f"{sorted(both)} given as both uncertain and fixed")

    required: set[str] = set()
    for o in organ_list:
        required |= {f"{o.name}.{q}" for q in KINETICS[o.kinetics]}
        if f"{o.name}.mass_kg" not in uncertain:
            fixed.setdefault(f"{o.name}.mass_kg", o.mass_ref_kg)
    allowed = required | {f"{o.name}.mass_kg" for o in organ_list} | {"administered_activity_mbq"}

    scale_names: dict[tuple[str, str | None], list[str]] = {}
    unknown = []
    for q in sorted((uncertain | set(fixed)) - allowed):
        cls = _classify(q, names)
        if cls is None:
            unknown.append(q)
        else:
            scale_names.setdefault(cls, []).append(q)
    if unknown:
        raise ValueError(f"unknown quantities for this model: {unknown}")
    missing = required - uncertain - set(fixed)
    if missing:
        raise ValueError(f"missing quantities (give as parameter or fixed): {sorted(missing)}")

    cross: dict[str, dict[str, float]] = {}
    for target, row in (cross_s_gy_per_mbq_h or {}).items():
        if target not in by_name:
            raise ValueError(f"cross_s: unknown target organ {target!r}")
        for source, value in row.items():
            if source not in by_name:
                raise ValueError(f"cross_s: unknown source organ {source!r}")
            if source == target:
                raise ValueError(f"cross_s: {target!r} <- {source!r} is the self term; use organs")
            if value < 0:
                raise ValueError("cross_s values must be >= 0")
        cross[target] = {s: float(v) for s, v in row.items()}

    if dataset is not None and cross_dose:
        regions = {o.name: o.region for o in organ_list if o.region}
        for target, row in dataset.cross_table(regions).items():
            for source, value in row.items():
                if source in cross.get(target, {}):
                    raise ValueError(
                        f"cross-organ S({target} <- {source}) is given both explicitly and by "
                        "the dataset"
                    )
                cross.setdefault(target, {})[source] = float(value)

    ratio_defs = []
    for r in ratios or []:
        extra = set(r) - {"name", "numerator", "denominator"}
        if extra or not {"name", "numerator", "denominator"} <= set(r):
            raise ValueError("each ratio needs exactly name, numerator, denominator")
        if r["numerator"] not in by_name or r["denominator"] not in by_name:
            raise ValueError(f"ratio {r['name']!r} refers to an unknown organ")
        if r["name"] in by_name:
            raise ValueError(f"ratio name {r['name']!r} clashes with an organ")
        ratio_defs.append((r["name"], r["numerator"], r["denominator"]))

    activity = fixed["administered_activity_mbq"]
    act_global = scale_names.get(("activity_scale", None), [])
    dose_global = scale_names.get(("dose_scale", None), [])

    def func(x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        n = len(next(iter(x.values())))

        def get(key: str):
            return x[key] if key in x else fixed[key]

        def product(keys: Sequence[str]):
            out = 1.0
            for k in keys:
                out = out * get(k)
            return out

        glob_act = product(act_global)
        tia = {}
        for o in organ_list:
            if o.kinetics == "mono_exponential":
                base = (
                    get(f"{o.name}.uptake_fraction")
                    * activity
                    * get(f"{o.name}.effective_half_life_h")
                    / LN2
                )
            else:
                base = get(f"{o.name}.tia_coefficient_h") * activity
            tia[o.name] = base * glob_act * product(scale_names.get(("activity_scale", o.name), []))

        glob_dose = product(dose_global)
        dose = {}
        for t in organ_list:
            s_self = t.s_self_gy_per_mbq_h * (get(f"{t.name}.mass_kg") / t.mass_ref_kg) ** (
                -t.mass_exponent
            )
            d = s_self * tia[t.name]
            for source, s_value in cross.get(t.name, {}).items():
                d = d + s_value * tia[source]
            dose[t.name] = d * glob_dose * product(scale_names.get(("dose_scale", t.name), []))

        out = {k: np.broadcast_to(v, (n,)) for k, v in dose.items()}
        for rname, num, den in ratio_defs:
            out[rname] = np.broadcast_to(dose[num] / dose[den], (n,))
        return out

    return UncertainModel(
        name=name or "multi-organ MIRD dose",
        inputs=InputSpace(parameters, correlations),
        func=func,
        outputs=[*names, *(r[0] for r in ratio_defs)],
        output_unit="Gy",
        output_units={r[0]: "" for r in ratio_defs},
        metadata={
            "model": "multi_organ",
            "nuclide": nuclide,
            "fixed": fixed,
            "organs": {o.name: o for o in organ_list},
            "cross_s_gy_per_mbq_h": cross,
            "s_values": None
            if dataset is None
            else {
                "dataset": dataset.dataset,
                "fingerprint": dataset.fingerprint,
                "cross_dose": cross_dose,
                "citation": dataset.provenance.get("citation"),
            },
        },
    )


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., UncertainModel]
    options: frozenset[str]  # extra top-level config keys the builder accepts


MODEL_BUILDERS: dict[str, ModelSpec] = {
    "multi_organ": ModelSpec(
        multi_organ,
        frozenset(
            {"organs", "cross_s_gy_per_mbq_h", "ratios", "nuclide", "s_values", "cross_dose"}
        ),
    ),
}
