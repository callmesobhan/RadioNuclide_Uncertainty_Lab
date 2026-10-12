"""One call from model to complete analysis: propagation, screening, Sobol', budgets."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy

from .budget import BudgetMatrix, ErrorBudget, budget_matrix, error_budget
from .config import RunSettings
from .diagnostics import Diagnostic, diagnose
from .model import UncertainModel
from .propagation import Summary, UncertaintyResult, propagate
from .sensitivity import RankSensitivity, SobolResult, rank_sensitivity, sobol_all


@dataclass(frozen=True)
class Analysis:
    model: UncertainModel
    run: RunSettings
    result: UncertaintyResult
    summaries: dict[str, Summary]
    ranks: dict[str, RankSensitivity]
    sobol: dict[str, SobolResult] | None
    budgets: dict[str, ErrorBudget] | None
    matrix: BudgetMatrix | None
    provenance: dict[str, Any]
    diagnostics: tuple[Diagnostic, ...] = ()
    data_integrity: dict[str, Any] | None = None


def analyze(
    model: UncertainModel,
    run: RunSettings | None = None,
    sobol: bool = True,
    config_sha256: str | None = None,
    data_integrity: dict[str, Any] | None = None,
) -> Analysis:
    """Run the full analysis described by ``run`` (defaults to :class:`RunSettings`)."""
    run = run or RunSettings()
    result = propagate(model, n=run.n, method=run.method, seed=run.seed)
    summaries = result.summaries(run.level)
    ranks = {o: rank_sensitivity(result, o) for o in model.outputs}

    sob = budgets = matrix = None
    if sobol:
        sob = sobol_all(model, n=run.sobol_n, seed=run.seed, conf=run.level)
        budgets = {o: error_budget(s, model.unit(o)) for o, s in sob.items()}
        matrix = budget_matrix(sob)

    prov = {
        "package": "RadioNuclide_Uncertainty_Lab",
        "version": _version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "config_sha256": config_sha256,
        "seed": run.seed,
        "sampling": result.method,
        "n_samples": result.n,
        "sobol_base_n": next(iter(sob.values())).n if sob else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    diags = diagnose(model, run, result, summaries, ranks, sob)
    return Analysis(
        model, run, result, summaries, ranks, sob, budgets, matrix, prov, diags, data_integrity
    )


def _version() -> str:
    from . import __version__

    return __version__
