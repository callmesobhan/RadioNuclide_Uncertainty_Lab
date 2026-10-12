"""Marginal distributions for uncertain inputs.

Every constructor returns a *frozen* ``scipy.stats`` distribution, so anything scipy offers
can also be passed straight to ``inputs.Parameter``. The constructors here cover
what dosimetry inputs usually need and encode the parametrisation people actually quote:

* ``lognormal(median, gsd)``  - positive quantities with multiplicative error (uptake,
  half-lives from fits, calibration factors). ``gsd`` is the geometric standard deviation.
* ``normal(mean, sd)`` / ``normal(mean, rel_sd=...)`` - symmetric, additive error.
* ``truncated_normal`` - normal restricted to a physical range (e.g. a fraction in [0, 1]).
* ``uniform`` / ``triangular`` - bounded ranges where only limits (and a mode) are known.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen


def normal(mean: float, sd: float | None = None, *, rel_sd: float | None = None) -> rv_frozen:
    """Normal distribution; give either an absolute ``sd`` or a relative ``rel_sd``."""
    sd = _resolve_sd(mean, sd, rel_sd)
    return stats.norm(loc=mean, scale=sd)


def truncated_normal(
    mean: float,
    sd: float | None = None,
    *,
    rel_sd: float | None = None,
    low: float = -math.inf,
    high: float = math.inf,
) -> rv_frozen:
    """Normal distribution truncated to ``[low, high]`` (``mean``/``sd`` refer to the parent)."""
    sd = _resolve_sd(mean, sd, rel_sd)
    if not low < high:
        raise ValueError(f"truncated_normal needs low < high, got low={low}, high={high}")
    return stats.truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)


def lognormal(median: float, gsd: float) -> rv_frozen:
    """Log-normal given its median and geometric standard deviation (``gsd`` > 1)."""
    if median <= 0:
        raise ValueError(f"lognormal median must be > 0, got {median}")
    if gsd <= 1:
        raise ValueError(f"lognormal gsd must be > 1, got {gsd}")
    return stats.lognorm(s=math.log(gsd), scale=median)


def uniform(low: float, high: float) -> rv_frozen:
    if not low < high:
        raise ValueError(f"uniform needs low < high, got low={low}, high={high}")
    return stats.uniform(loc=low, scale=high - low)


def triangular(low: float, mode: float, high: float) -> rv_frozen:
    if not low <= mode <= high or low == high:
        raise ValueError(f"triangular needs low <= mode <= high, got {low}, {mode}, {high}")
    return stats.triang(c=(mode - low) / (high - low), loc=low, scale=high - low)


def _resolve_sd(mean: float, sd: float | None, rel_sd: float | None) -> float:
    if (sd is None) == (rel_sd is None):
        raise ValueError("give exactly one of sd or rel_sd")
    value = sd if sd is not None else abs(mean) * rel_sd  # type: ignore[operator]
    if value <= 0:
        raise ValueError(f"standard deviation must be > 0, got {value}")
    return float(value)


# --- config-file support -----------------------------------------------------------------------

_SPEC_KEYS: dict[str, set[str]] = {
    "normal": {"mean", "sd", "rel_sd", "low", "high"},  # low/high => truncated
    "lognormal": {"median", "gsd"},
    "uniform": {"low", "high"},
    "triangular": {"low", "mode", "high"},
}


def from_spec(spec: Mapping[str, Any]) -> rv_frozen:
    """Build a distribution from a config mapping.

    Example: ``{dist: lognormal, median: 0.04, gsd: 1.2}``. Unknown keys are rejected so
    typos in a config cannot silently change the physics.
    """
    spec = dict(spec)
    kind = spec.pop("dist", None)
    if kind not in _SPEC_KEYS:
        raise ValueError(f"unknown dist {kind!r}; choose from {sorted(_SPEC_KEYS)}")
    extra = set(spec) - _SPEC_KEYS[kind]
    if extra:
        raise ValueError(f"dist {kind!r} does not accept {sorted(extra)}")
    try:
        if kind == "normal":
            if "low" in spec or "high" in spec:
                return truncated_normal(
                    spec["mean"],
                    spec.get("sd"),
                    rel_sd=spec.get("rel_sd"),
                    low=spec.get("low", -math.inf),
                    high=spec.get("high", math.inf),
                )
            return normal(spec["mean"], spec.get("sd"), rel_sd=spec.get("rel_sd"))
        if kind == "lognormal":
            return lognormal(spec["median"], spec["gsd"])
        if kind == "uniform":
            return uniform(spec["low"], spec["high"])
        return triangular(spec["low"], spec["mode"], spec["high"])
    except KeyError as err:
        raise ValueError(f"dist {kind!r} is missing required key {err.args[0]!r}") from None
