"""Published-number registry and consistency checks that exclude conflicting data.

Why: transcribing numbers from papers goes wrong in three ways. A source can contradict itself
(an abstract that swaps a mean and a median), two sources can disagree, or a number can be copied
wrongly into code. This module makes all three detectable by machine:

* every published number is a :class:`Claim` with its location (``data/claims.yaml``);
* :class:`Check` objects relate claims to each other, to arithmetic, or to physics (reciprocity,
  energy conservation) and to the packaged reference datasets;
* a check that fails *excludes* the claims it names: :meth:`ClaimRegistry.require` then raises
  :class:`ExcludedClaimError`, and a config parameter that points at such a claim refuses to load;
* a discrepancy we already know about is declared ``expect: conflict``. It must keep failing, so
  if the source is corrected (or our reading was wrong) the check reports ``UNEXPECTED_PASS`` and
  CI forces someone to look.

Run it with ``radionuclide-uncertainty-lab check-data``.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

PASS, FAIL, KNOWN_CONFLICT, UNEXPECTED_PASS = "PASS", "FAIL", "KNOWN_CONFLICT", "UNEXPECTED_PASS"


class ExcludedClaimError(ValueError):
    """Raised when excluded data is requested."""


@dataclass(frozen=True)
class Claim:
    id: str
    value: float
    unit: str
    source: str
    locator: str


@dataclass(frozen=True)
class CheckResult:
    id: str
    kind: str
    status: str
    detail: str
    note: str = ""
    excluded: tuple[str, ...] = ()
    dataset: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (PASS, KNOWN_CONFLICT)


# --------------------------------------------------------------------------------------------
# Safe expression evaluation (no eval): numbers, claim names, + - * / **, sqrt/abs/log
# --------------------------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "log": math.log}
_CLAIM_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+")


def evaluate_expression(expr: str, value_of) -> float:
    """Evaluate ``expr`` where dotted names are claim ids resolved through ``value_of``."""
    names: dict[str, str] = {}

    def alias(match: re.Match[str]) -> str:
        key = f"_c{len(names)}"
        names[key] = match.group(0)
        return key

    tree = ast.parse(_CLAIM_TOKEN.sub(alias, expr), mode="eval")

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return value_of(names[node.id])
            raise ValueError(f"unknown name {node.id!r} in expression {expr!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large in expression")
            return _BINOPS[type(node.op)](left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and len(node.args) == 1
            and not node.keywords
        ):
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise ValueError(f"disallowed construct {type(node).__name__} in expression {expr!r}")

    return ev(tree)


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------


class ClaimRegistry:
    def __init__(
        self,
        claims: Mapping[str, Claim],
        checks: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Any] | None = None,
    ) -> None:
        self.claims = dict(claims)
        self.sources = dict(sources or {})
        self._checks = list(checks)
        self._results: list[CheckResult] | None = None

    # -- access ---------------------------------------------------------------------------

    def get(self, claim_id: str) -> Claim:
        try:
            return self.claims[claim_id]
        except KeyError:
            raise KeyError(f"unknown claim {claim_id!r}") from None

    def require(self, claim_id: str) -> Claim:
        """The claim, unless a failed check excluded it."""
        claim = self.get(claim_id)
        reason = self.excluded.get(claim_id)
        if reason:
            raise ExcludedClaimError(f"claim {claim_id!r} is excluded: {reason}")
        return claim

    def value(self, claim_id: str) -> float:
        return self.require(claim_id).value

    # -- checks ---------------------------------------------------------------------------

    @property
    def results(self) -> list[CheckResult]:
        if self._results is None:
            self._results = [self._run(c) for c in self._checks]
        return self._results

    @property
    def excluded(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in self.results:
            for cid in r.excluded:
                out.setdefault(cid, f"{r.id}: {r.detail}" + (f" ({r.note})" if r.note else ""))
        return out

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def _operand(self, op: Any) -> float:
        if isinstance(op, int | float):
            return float(op)
        if isinstance(op, str):
            return self.get(op).value
        if isinstance(op, Mapping) and "expr" in op:
            return evaluate_expression(op["expr"], lambda cid: self.get(cid).value)
        raise ValueError(f"bad operand {op!r}")

    @staticmethod
    def _named_claims(spec: Mapping[str, Any]) -> list[str]:
        """Claim ids a check names directly (not those inside expressions)."""
        out: list[str] = []
        for key in ("a", "b", "total", "value"):
            if isinstance(spec.get(key), str):
                out.append(spec[key])
        for key in ("terms", "values"):
            out += [v for v in spec.get(key, []) if isinstance(v, str)]
        return out

    @staticmethod
    def _plain_claims(*operands: Any) -> list[str]:
        return [o for o in operands if isinstance(o, str)]

    def _run(self, spec: Mapping[str, Any]) -> CheckResult:
        kind, cid = spec["kind"], spec["id"]
        expect = spec.get("expect", "pass")
        try:
            ok, detail, default_excl = self._evaluate(kind, spec)
        except (KeyError, ValueError, ZeroDivisionError) as err:
            # fail closed: if the check cannot run, the claims it was protecting are not trusted
            ok, detail = False, f"could not evaluate: {err}"
            default_excl = [c for c in self._named_claims(spec) if c in self.claims]
        if expect == "pass":
            status = PASS if ok else FAIL
        elif expect == "conflict":
            status = UNEXPECTED_PASS if ok else KNOWN_CONFLICT
        else:
            raise ValueError(f"check {cid!r}: expect must be 'pass' or 'conflict'")
        excluded = (
            tuple(spec.get("exclude", default_excl)) if status in (FAIL, KNOWN_CONFLICT) else ()
        )
        return CheckResult(
            cid, kind, status, detail, spec.get("note", ""), excluded, spec.get("dataset")
        )

    def _evaluate(self, kind: str, s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
        if kind == "agree":
            a, b = self._operand(s["a"]), self._operand(s["b"])
            if "abs_tol" in s:
                ok = abs(a - b) <= s["abs_tol"]
                tol = f"|diff| <= {s['abs_tol']}"
            else:
                ok = abs(a - b) <= s["rel_tol"] * max(abs(a), abs(b))
                tol = f"rel diff <= {s['rel_tol']:.3g}"
            return ok, f"{a:.6g} vs {b:.6g} ({tol})", self._plain_claims(s["a"], s["b"])
        if kind == "quadrature":
            terms = [self._operand(t) for t in s["terms"]]
            q, total = math.sqrt(sum(t * t for t in terms)), self._operand(s["total"])
            tol = s.get("tol", 1.0)
            ok = (
                (q - 1e-9 <= total <= q + tol)
                if s.get("mode", "nearest") == "up"
                else abs(total - q) <= tol
            )
            return (
                ok,
                f"quadrature of {len(terms)} terms = {q:.4g}; stated total {total:.4g}",
                self._plain_claims(s["total"]),
            )
        if kind == "bound":
            v = self._operand(s["value"])
            lo = self._operand(s["lo"]) if "lo" in s else -math.inf
            hi = self._operand(s["hi"]) if "hi" in s else math.inf
            return (
                lo <= v <= hi,
                f"{v:.6g} within [{lo:.6g}, {hi:.6g}]",
                self._plain_claims(s["value"]),
            )
        if kind == "spread":
            vals = [self._operand(v) for v in s["values"]]
            ratio = max(vals) / min(vals)
            return (
                ratio <= s["max_ratio"],
                f"values {[round(v, 4) for v in vals]}, max/min = {ratio:.3g} "
                f"(limit {s['max_ratio']})",
                self._plain_claims(*s["values"]),
            )
        if kind == "refdata":
            return _REFDATA_CHECKS[s["name"]](s)
        raise ValueError(f"unknown check kind {kind!r}")

    # -- reporting ------------------------------------------------------------------------

    def format_report(self) -> str:
        lines = [f"{'STATUS':<16} {'CHECK':<46} DETAIL"]
        for r in self.results:
            lines.append(f"{r.status:<16} {r.id:<46} {r.detail}")
        if self.excluded:
            lines += ["", "Excluded claims (will not be used):"]
            lines += [f"  {cid}  <- {why}" for cid, why in sorted(self.excluded.items())]
        counts = {
            s: sum(r.status == s for r in self.results)
            for s in (PASS, KNOWN_CONFLICT, FAIL, UNEXPECTED_PASS)
        }
        lines += [
            "",
            f"{len(self.claims)} claims, {len(self.results)} checks: "
            + ", ".join(f"{n} {s}" for s, n in counts.items() if n),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Reference-dataset checks
# --------------------------------------------------------------------------------------------


def _dataset(s: Mapping[str, Any]):
    from .refdata import cached_svalues

    return cached_svalues(s["dataset"])


def _check_reciprocity(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Specific absorbed fractions are symmetric, S(t<-s) = S(s<-t), up to Monte Carlo noise.

    That noise grows as a cross term shrinks (roughly as 1/sqrt of its size, because fewer particle
    histories contribute), so a *relative* asymmetry of a tiny cross term says nothing. With
    ``min_relative`` only pairs whose mean cross S is at least that fraction of the geometric mean
    of the two self S-values are tested; the detail says how many pairs were tested and skipped.
    ``regions`` restricts the test to the named regions (e.g. the organs a scenario uses).
    """
    import math

    ds, tol = _dataset(s), s.get("rel_tol", 0.05)
    floor = s.get("min_relative", 0.0)
    worst, where, tested, skipped = 0.0, "", 0, 0
    names = [r for r in ds.regions if r in s["regions"]] if "regions" in s else list(ds.regions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            x, y = ds.s(a, b), ds.s(b, a)
            mean = (x + y) / 2
            if mean <= 0 or mean / math.sqrt(ds.self_s(a) * ds.self_s(b)) < floor:
                skipped += 1
                continue
            tested += 1
            d = abs(x - y) / mean
            if d > worst:
                worst, where = d, f"{a} <-> {b}"
    if tested == 0:
        return False, "no pair above the noise floor was tested", []
    return (
        worst <= tol,
        f"{tested} pairs tested ({skipped} below the noise floor {floor:g} skipped); largest "
        f"asymmetry {worst:.2%} ({where}); limit {tol:.0%}",
        [],
    )


def _check_energy_conservation(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    from .mird import DECAYS_PER_MBQ_H, MEV_TO_J
    from .nuclides import get_nuclide

    ds = _dataset(s)
    nuc = get_nuclide(ds.nuclide)
    emitted = (nuc.delta_e_mev + nuc.delta_ph_mev) * MEV_TO_J * DECAYS_PER_MBQ_H  # J per MBq h
    worst, where = 0.0, ""
    for src in ds.regions:
        absorbed = sum(ds.s(t, src) * ds.mass(t) for t in ds.regions)  # J per MBq h
        frac = absorbed / emitted
        if frac > worst:
            worst, where = frac, src
    return (
        worst <= 1.0,
        f"largest absorbed/emitted = {worst:.3f} (source {where}); must be <= 1",
        [],
    )


def _check_matches_claims(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    ds, tol = _dataset(s), s.get("rel_tol", 0.01)
    registry = get_registry()
    worst, where = 0.0, ""
    for region, ids in s["pairs"].items():
        s_pub = registry.get(ids["s"]).value * 1e-3  # mGy/(MBq h) -> Gy/(MBq h)
        m_pub = registry.get(ids["mass"]).value * 1e-3  # g -> kg
        for label, mine, pub in (("S", ds.self_s(region), s_pub), ("mass", ds.mass(region), m_pub)):
            d = abs(mine - pub) / pub
            if d > worst:
                worst, where = d, f"{region} {label}"
    return worst <= tol, f"largest deviation {worst:.2%} ({where}); limit {tol:.0%}", []


MGY_PER_MBQ_S_PER_GY_PER_MBQ_H = 1000.0 / 3600.0  # Gy/(MBq h) -> mGy/(MBq s)


def _check_matches_s_claims(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """S(target <- source) of a dataset against published S-values, in mGy/(MBq s)."""
    ds, tol = _dataset(s), s.get("rel_tol", 0.01)
    registry = get_registry()
    worst, where = 0.0, ""
    for pair in s["pairs"]:
        mine = ds.s(pair["target"], pair["source"]) * MGY_PER_MBQ_S_PER_GY_PER_MBQ_H
        pub = registry.get(pair["claim"]).value
        d = abs(mine - pub) / pub
        if d > worst:
            worst, where = d, f"{pair['target']} <- {pair['source']}"
    return (
        worst <= tol,
        f"{len(s['pairs'])} pairs, largest deviation {worst:.2%} ({where}); limit {tol:.2%}",
        [],
    )


def _check_anchored_cross_share(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Share of one organ's anchored dose that comes from the other organs, against a published
    statement (the EANM recommendations: about 2 % of the kidney dose from the rest of the body)."""
    import numpy as np

    from .mird import tia_coefficients_from_doses

    ds, registry = _dataset(s), get_registry()
    organs = list(s["organs"])  # model organ -> {region, dose claim}
    s_matrix = np.array(
        [[ds.s(s["organs"][t]["region"], s["organs"][u]["region"]) for u in organs] for t in organs]
    )
    doses = np.array([registry.get(s["organs"][o]["dose"]).value for o in organs])
    tia = tia_coefficients_from_doses(s_matrix, doses)
    k = organs.index(s["target"])
    total = float(s_matrix[k] @ tia)
    share = 100.0 * (total - s_matrix[k, k] * tia[k]) / total
    pub = registry.get(s["claim"]).value
    return (
        abs(share - pub) <= s["abs_tol"],
        f"{s['target']}: {share:.2f} % of the dose from other organs; published {pub:g} % "
        f"(+-{s['abs_tol']} pp)",
        [],
    )


def _check_nuclide_registry(s: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Values in the nuclide registry must equal the claims they cite (no stale copies)."""
    from .nuclides import get_nuclide

    nuc, registry = get_nuclide(s["nuclide"]), get_registry()
    tol = s.get("rel_tol", 1e-9)
    bad = []
    for attr, claim_id in s["fields"].items():
        mine, pub = float(getattr(nuc, attr)), registry.get(claim_id).value
        if abs(mine - pub) > tol * max(abs(pub), 1e-300):
            bad.append(f"{attr}={mine:g} but {claim_id}={pub:g}")
    return not bad, "; ".join(bad) or f"{len(s['fields'])} registry fields equal their claims", []


_REFDATA_CHECKS = {
    "reciprocity": _check_reciprocity,
    "energy_conservation": _check_energy_conservation,
    "matches_claims": _check_matches_claims,
    "matches_s_claims": _check_matches_s_claims,
    "anchored_cross_share": _check_anchored_cross_share,
    "nuclide_registry": _check_nuclide_registry,
}


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def load_registry(text: str | None = None) -> ClaimRegistry:
    raw = yaml.safe_load(
        text
        if text is not None
        else (resources.files(__package__) / "data" / "claims.yaml").read_text(encoding="utf-8")
    )
    claims = {}
    for cid, d in raw["claims"].items():
        extra = set(d) - {"value", "unit", "source", "locator"}
        if extra:
            raise ValueError(f"claim {cid!r}: unknown keys {sorted(extra)}")
        if d["source"] not in raw["sources"]:
            raise ValueError(f"claim {cid!r}: unknown source {d['source']!r}")
        claims[cid] = Claim(cid, float(d["value"]), d["unit"], d["source"], d["locator"])
    ids = [c["id"] for c in raw["checks"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate check ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return ClaimRegistry(claims, raw["checks"], raw["sources"])


@lru_cache(maxsize=1)
def get_registry() -> ClaimRegistry:
    return load_registry()


def require_trusted_dataset(name: str) -> None:
    """Refuse a reference dataset unless it has checks and none of them fails.

    A *declared* known conflict (``expect: conflict``) does not block: it records a discrepancy with
    an external claim that has already been excluded. A failing check, or a known conflict that has
    disappeared (``UNEXPECTED_PASS``, so the record is stale), does block."""
    mine = [r for r in get_registry().results if r.dataset == name]
    if not mine:
        raise ValueError(
            f"dataset {name!r} has no registered consistency checks; refusing to use it"
        )
    bad = [r for r in mine if r.status in (FAIL, UNEXPECTED_PASS)]
    if bad:
        raise ValueError(
            f"dataset {name!r} failed consistency checks: "
            + "; ".join(f"{r.id} [{r.status}] {r.detail}" for r in bad)
        )


# --------------------------------------------------------------------------------------------
# Tie a config entry to a claim
# --------------------------------------------------------------------------------------------


def _require_for(what: str, claim_id: str) -> Claim:
    try:
        return get_registry().require(claim_id)
    except KeyError:
        raise ValueError(f"{what}: unknown claim {claim_id!r} (see data/claims.yaml)") from None


def verify_parameter_claim(name: str, dist, claim_id: str, rel_tol: float = 0.03) -> None:
    """A parameter that cites a claim must realise it: relative SD of the distribution (percent)
    equals the claim. Excluded claims are refused."""
    claim = _require_for(f"parameter {name!r}", claim_id)
    mean = float(dist.mean())
    if mean == 0:
        raise ValueError(
            f"parameter {name!r}: cannot compare a relative SD for a zero-mean distribution"
        )
    rel_sd = float(dist.std()) / abs(mean) * 100
    if abs(rel_sd - claim.value) > max(rel_tol * claim.value, 0.02):
        raise ValueError(
            f"parameter {name!r}: relative SD {rel_sd:.3g} % does not match claim "
            f"{claim_id} = {claim.value:g} % ({claim.locator})"
        )


def verify_correlation_claim(
    pair: tuple[str, str], rho: float, claim_id: str, abs_tol: float = 0.01
) -> None:
    claim = _require_for(f"correlation {pair}", claim_id)
    if abs(rho - claim.value) > abs_tol:
        raise ValueError(
            f"correlation {pair}: rho {rho:g} does not match claim {claim_id} = {claim.value:g}"
        )
