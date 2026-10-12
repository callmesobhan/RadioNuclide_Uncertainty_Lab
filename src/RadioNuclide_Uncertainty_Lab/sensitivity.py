"""Sensitivity analysis: cheap rank-based screening and variance-based (Sobol') indices.

Two tools, deliberately kept separate:

* :func:`rank_sensitivity` reuses an existing Monte Carlo run (no extra model evaluations).
  Good for a first ranking and for monotonic models; it treats parameters individually, even
  inside a correlated group.
* :func:`sobol_indices` needs ``n * (2 + K)`` model evaluations for ``K`` factors, but gives
  a proper additive decomposition of the output variance, including interactions, per
  *factor* (see :mod:`RadioNuclide_Uncertainty_Lab.inputs`). The error budget is built on this.
  For a multi-output model the design is evaluated once and every output is analysed from it
  (:func:`sobol_all`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from .model import UncertainModel
from .propagation import UncertaintyResult
from .sampling import unit_samples

# --------------------------------------------------------------------------------------------
# Rank-based screening
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankSensitivity:
    output: str
    names: tuple[str, ...]
    spearman: np.ndarray  # rank correlation of each input with the output
    prcc: np.ndarray  # partial rank correlation (other inputs held fixed)
    src: np.ndarray  # standardised regression coefficients on the raw values
    r2: float  # R^2 of the linear regression behind ``src``; low => non-linear/interacting

    def ranking(self, by: str = "prcc") -> list[tuple[str, float]]:
        """Parameters sorted by decreasing ``|by|`` (``spearman``, ``prcc`` or ``src``)."""
        values = getattr(self, by)
        order = np.argsort(-np.abs(values))
        return [(self.names[i], float(values[i])) for i in order]


def rank_sensitivity(result: UncertaintyResult, output: str | None = None) -> RankSensitivity:
    name = result._resolve(output)
    names = tuple(result.inputs)
    X = np.column_stack([result.inputs[k] for k in names])
    y = result[name]
    if np.ptp(y) == 0:
        raise ValueError(f"output {name!r} is constant; sensitivity is undefined")

    # parameters that never vary (e.g. a degenerate distribution) have no defined correlation
    varies = X.std(0) > 0
    Rx = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    Ry = rankdata(y)
    spearman = np.array([_corr(Rx[:, j], Ry) if varies[j] else 0.0 for j in range(X.shape[1])])

    prcc = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if not varies[j]:
            continue
        others = np.delete(Rx, j, axis=1)
        prcc[j] = _corr(_residual(Rx[:, j], others), _residual(Ry, others))

    Z = np.where(varies, (X - X.mean(0)) / np.where(varies, X.std(0, ddof=1), 1.0), 0.0)
    zy = (y - y.mean()) / y.std(ddof=1)
    design = np.column_stack([np.ones(len(zy)), Z])
    coef, *_ = np.linalg.lstsq(design, zy, rcond=None)
    pred = design @ coef
    r2 = 1.0 - float(np.sum((zy - pred) ** 2) / np.sum(zy**2))

    return RankSensitivity(name, names, spearman, prcc, coef[1:], r2)


def _residual(v: np.ndarray, others: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(v)), others])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ coef


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------------------------
# Variance-based (Sobol') indices
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SobolResult:
    """First-order (``S1``) and total-order (``ST``) indices per factor, for one output.

    ``S1``: share of output variance explained by the factor alone.
    ``ST``: share of output variance that disappears if every other factor were fixed, i.e.
    S1 plus every interaction term the factor takes part in. In theory ``ST >= S1`` and
    ``sum(S1) <= 1`` (equality for additive models); estimates can stray slightly past
    these bounds by sampling noise.
    Confidence intervals are percentile bootstrap intervals over the sample rows.
    """

    output: str
    factors: tuple[str, ...]
    members: tuple[tuple[str, ...], ...]
    s1: np.ndarray
    st: np.ndarray
    s1_ci: np.ndarray  # shape (K, 2)
    st_ci: np.ndarray  # shape (K, 2)
    mean: float
    variance: float
    n: int  # base sample size; total model evaluations = n * (2 + K)
    conf: float
    seed: int | None

    @property
    def n_evaluations(self) -> int:
        return self.n * (2 + len(self.factors))

    @property
    def interaction_share(self) -> float:
        """Variance not explained by any single factor acting alone: ``1 - sum(S1)``."""
        return float(1.0 - self.s1.sum())


def sobol_all(
    model: UncertainModel,
    n: int = 2**13,
    seed: int | None = None,
    n_boot: int = 200,
    conf: float = 0.95,
) -> dict[str, SobolResult]:
    """Saltelli (2010) first-order and Jansen (1999) total-order estimators, all outputs.

    Factors are the groups of ``model.inputs``; each factor is resampled as a block, so
    correlation *within* a factor is respected (and independence *between* factors holds).
    ``n`` is rounded up to a power of two. Outputs with zero variance are reported with an
    error, as their indices are undefined.
    """
    d = model.inputs.dim
    factors = model.inputs.factors
    ab = unit_samples(n, 2 * d, method="sobol", seed=seed)
    n = len(ab)
    A, B = ab[:, :d], ab[:, d:]

    f_a = model.evaluate(A)[1]  # (n, m)
    f_b = model.evaluate(B)[1]
    f_ab = np.empty((len(factors), n, len(model.outputs)))
    for k, f in enumerate(factors):
        mixed = A.copy()
        cols = list(f.columns)
        mixed[:, cols] = B[:, cols]
        f_ab[k] = model.evaluate(mixed)[1]

    rng = np.random.default_rng(None if seed is None else seed + 1)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    q = [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100]

    results: dict[str, SobolResult] = {}
    for j, name in enumerate(model.outputs):
        a, b, ab_j = f_a[:, j], f_b[:, j], f_ab[:, :, j]
        try:
            s1, st = _estimators(a, b, ab_j)
        except ValueError:
            raise ValueError(
                f"output {name!r} has zero variance; sensitivity is undefined"
            ) from None
        boot_s1 = np.empty((n_boot, len(factors)))
        boot_st = np.empty_like(boot_s1)
        for i, idx in enumerate(boot_idx):
            boot_s1[i], boot_st[i] = _estimators(a[idx], b[idx], ab_j[:, idx])
        both = np.concatenate([a, b])
        results[name] = SobolResult(
            output=name,
            factors=tuple(f.name for f in factors),
            members=tuple(f.members for f in factors),
            s1=s1,
            st=st,
            s1_ci=np.percentile(boot_s1, q, axis=0).T,
            st_ci=np.percentile(boot_st, q, axis=0).T,
            mean=float(both.mean()),
            variance=float(both.var(ddof=1)),
            n=n,
            conf=conf,
            seed=seed,
        )
    return results


def sobol_indices(
    model: UncertainModel,
    n: int = 2**13,
    seed: int | None = None,
    n_boot: int = 200,
    conf: float = 0.95,
    output: str | None = None,
) -> SobolResult:
    """Sobol' indices of one output (``None`` allowed for single-output models).

    The whole design is still evaluated once; use :func:`sobol_all` to get every output.
    """
    name = model.resolve_output(output)
    return sobol_all(model, n=n, seed=seed, n_boot=n_boot, conf=conf)[name]


def _estimators(
    f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([f_a, f_b])
    var = both.var(ddof=1)
    if var == 0:
        raise ValueError("output variance is zero; sensitivity is undefined")
    # Centering leaves the estimator's expectation unchanged (E[f_ab] = E[f_a]) but removes
    # the large mean-times-noise term, which matters when |mean| >> SD (typical for doses).
    f_b = f_b - both.mean()
    s1 = np.mean(f_b * (f_ab - f_a), axis=1) / var
    st = 0.5 * np.mean((f_a - f_ab) ** 2, axis=1) / var
    return s1, st
