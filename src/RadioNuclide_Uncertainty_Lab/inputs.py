"""Uncertain inputs: parameters, groups (factors) and correlation.

Design
------
* A :class:`Parameter` is one uncertain quantity with a marginal distribution, a unit, a
  *group* and a free-text ``source`` recording where the uncertainty came from.
* A **factor** is the unit the sensitivity analysis works on: all parameters that share a
  ``group`` form one factor; a parameter with ``group=None`` is its own factor. Factors are
  statistically independent of each other.
* Correlation is only allowed **inside** a factor. That is what keeps variance-based
  (Sobol') indices well defined: a correlated block is varied as one unit, and the error
  budget reports it as one line (e.g. "biokinetic fit") instead of pretending the correlated
  parameters can be ranked separately.
* Correlation is imposed with a Gaussian copula. The stated ``rho`` is the correlation of the
  latent normal scores, not the Pearson correlation of the transformed marginals (they agree
  for normal marginals). See ``docs/methodology.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_frozen

_EPS = 1e-12


@dataclass(frozen=True)
class Parameter:
    name: str
    dist: rv_frozen
    unit: str = ""
    group: str | None = None
    description: str = ""
    source: str = ""
    claim: str = ""  # id of the published claim this uncertainty realises


@dataclass(frozen=True)
class Factor:
    """A set of parameters varied together in the sensitivity analysis."""

    name: str
    columns: tuple[int, ...]
    members: tuple[str, ...]


class InputSpace:
    """The joint input distribution: independent factors, optional correlation within one."""

    def __init__(
        self,
        parameters: Sequence[Parameter],
        correlations: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        if not parameters:
            raise ValueError("an InputSpace needs at least one uncertain parameter")
        names = [p.name for p in parameters]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate parameter names: {dupes}")
        self.parameters = tuple(parameters)
        self.names = tuple(names)
        self.correlations = dict(correlations or {})
        self.factors = self._build_factors()
        self._copula = self._build_copula()

    # -- construction ---------------------------------------------------------------------

    def _build_factors(self) -> tuple[Factor, ...]:
        order: dict[str, list[int]] = {}
        for col, p in enumerate(self.parameters):
            key = p.group if p.group is not None else f"\0{p.name}"
            order.setdefault(key, []).append(col)
        return tuple(
            Factor(
                name=self.parameters[cols[0]].name if key.startswith("\0") else key,
                columns=tuple(cols),
                members=tuple(self.names[c] for c in cols),
            )
            for key, cols in order.items()
        )

    def _build_copula(self) -> list[tuple[list[int], np.ndarray]]:
        """Cholesky factors for each factor that has correlation specified."""
        index = {n: i for i, n in enumerate(self.names)}
        by_factor: dict[int, list[tuple[int, int, float]]] = {}
        for (a, b), rho in self.correlations.items():
            if a not in index or b not in index:
                raise ValueError(f"correlation refers to unknown parameter in {(a, b)}")
            if a == b:
                raise ValueError(f"correlation of {a!r} with itself")
            if not -1 < rho < 1:
                raise ValueError(f"rho for {(a, b)} must be in (-1, 1), got {rho}")
            fa, fb = self._factor_of(index[a]), self._factor_of(index[b])
            if fa != fb or self.parameters[index[a]].group is None:
                raise ValueError(
                    f"correlated parameters {a!r} and {b!r} must share a group so they are "
                    "treated as one factor in the sensitivity analysis"
                )
            by_factor.setdefault(fa, []).append((index[a], index[b], rho))

        blocks = []
        for fi, pairs in by_factor.items():
            cols = list(self.factors[fi].columns)
            pos = {c: k for k, c in enumerate(cols)}
            corr = np.eye(len(cols))
            for ca, cb, rho in pairs:
                corr[pos[ca], pos[cb]] = corr[pos[cb], pos[ca]] = rho
            try:
                chol = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"correlation matrix of group {self.factors[fi].name!r} is not "
                    "positive definite"
                ) from None
            blocks.append((cols, chol))
        return blocks

    def _factor_of(self, column: int) -> int:
        for i, f in enumerate(self.factors):
            if column in f.columns:
                return i
        raise AssertionError("unreachable")

    # -- sampling -------------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def transform(self, u: np.ndarray) -> dict[str, np.ndarray]:
        """Map unit-cube samples ``(n, dim)`` to physical parameter values."""
        u = np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS)
        if u.ndim != 2 or u.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {u.shape}")
        if self._copula:
            u = u.copy()
            for cols, chol in self._copula:
                z = norm.ppf(u[:, cols]) @ chol.T
                u[:, cols] = norm.cdf(z)
        return {p.name: p.dist.ppf(u[:, i]) for i, p in enumerate(self.parameters)}

    def describe(self) -> list[dict[str, object]]:
        """One row per parameter: for reports and for checking a config by eye."""
        rows = []
        for p in self.parameters:
            rows.append(
                {
                    "name": p.name,
                    "group": p.group or "-",
                    "unit": p.unit,
                    "mean": float(p.dist.mean()),
                    "sd": float(p.dist.std()),
                    "median": float(p.dist.median()),
                    "source": p.source,
                    "claim": p.claim,
                }
            )
        return rows
