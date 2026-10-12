"""Error budget: turn Sobol' indices into statements a physicist can put in a paper.

Each line reports, per factor (group of parameters):

* ``s1`` / ``st``  - first-order and total variance shares (dimensionless);
* ``sd_first`` / ``sd_total`` - the same shares expressed as a standard deviation of the
  output, ``sqrt(share * Var[Y])``, in output units;
* ``rel_first`` / ``rel_total`` - those standard deviations relative to the output mean.

Standard deviations of different factors do **not** add linearly; variances (the shares) do.
The budget therefore always reports the share first and the SD as a reading aid.

For multi-output models, :func:`budget_matrix` lays the total-order shares out as
factors x outputs, which shows at a glance which sources are shared between organs and which
belong to one organ only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .sensitivity import SobolResult


@dataclass(frozen=True)
class BudgetLine:
    factor: str
    members: tuple[str, ...]
    s1: float
    s1_ci: tuple[float, float]
    st: float
    st_ci: tuple[float, float]
    sd_first: float
    sd_total: float
    rel_first: float
    rel_total: float


@dataclass(frozen=True)
class ErrorBudget:
    output: str
    output_mean: float
    output_sd: float
    output_unit: str
    interaction_share: float
    lines: tuple[BudgetLine, ...]  # sorted by decreasing total-order index
    n_evaluations: int
    conf: float

    @property
    def relative_sd(self) -> float:
        return self.output_sd / abs(self.output_mean) if self.output_mean else math.inf


def error_budget(sobol: SobolResult, output_unit: str = "") -> ErrorBudget:
    mean, var = sobol.mean, sobol.variance
    lines = []
    for k, name in enumerate(sobol.factors):
        s1, st = float(sobol.s1[k]), float(sobol.st[k])
        # Negative estimates are sampling noise around ~0; clamp only for the SD conversion.
        sd1, sdt = math.sqrt(max(s1, 0.0) * var), math.sqrt(max(st, 0.0) * var)
        lines.append(
            BudgetLine(
                factor=name,
                members=sobol.members[k],
                s1=s1,
                s1_ci=(float(sobol.s1_ci[k, 0]), float(sobol.s1_ci[k, 1])),
                st=st,
                st_ci=(float(sobol.st_ci[k, 0]), float(sobol.st_ci[k, 1])),
                sd_first=sd1,
                sd_total=sdt,
                rel_first=sd1 / abs(mean) if mean else math.inf,
                rel_total=sdt / abs(mean) if mean else math.inf,
            )
        )
    lines.sort(key=lambda ln: ln.st, reverse=True)
    return ErrorBudget(
        output=sobol.output,
        output_mean=mean,
        output_sd=math.sqrt(var),
        output_unit=output_unit,
        interaction_share=sobol.interaction_share,
        lines=tuple(lines),
        n_evaluations=sobol.n_evaluations,
        conf=sobol.conf,
    )


def budget_markdown(budget: ErrorBudget, min_share: float = 0.0) -> str:
    """Markdown table. Factors whose total-order share is below ``min_share`` are folded into
    one summary row, so long budgets stay readable."""
    pct = int(round(budget.conf * 100))
    head = (
        f"| Factor | Parameters | S1 | {pct}% CI | ST | {pct}% CI | "
        f"SD (first) | rel. | SD (total) | rel. |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    shown = [ln for ln in budget.lines if ln.st >= min_share]
    folded = [ln for ln in budget.lines if ln.st < min_share]
    rows = [
        f"| {ln.factor} | {', '.join(ln.members)} | {ln.s1:.3f} | "
        f"[{ln.s1_ci[0]:.3f}, {ln.s1_ci[1]:.3f}] | {ln.st:.3f} | "
        f"[{ln.st_ci[0]:.3f}, {ln.st_ci[1]:.3f}] | {ln.sd_first:.3g} | "
        f"{ln.rel_first:.1%} | {ln.sd_total:.3g} | {ln.rel_total:.1%} |"
        for ln in shown
    ]
    if folded:
        total = sum(max(ln.st, 0.0) for ln in folded)
        rows.append(
            f"| *{len(folded)} other factors (each ST < {min_share:.1%})* | | | | "
            f"{total:.3f} | | | | | |"
        )
    foot = (
        f"\n\nOutput `{budget.output}`: mean {budget.output_mean:.4g} {budget.output_unit}, "
        f"SD {budget.output_sd:.3g} ({budget.relative_sd:.1%}). "
        f"Variance not explained by any single factor (interactions): "
        f"{budget.interaction_share:.1%}. Model evaluations: {budget.n_evaluations:,}."
    )
    return head + "\n".join(rows) + foot


@dataclass(frozen=True)
class BudgetMatrix:
    """Total- and first-order shares as ``(factors x outputs)`` arrays."""

    factors: tuple[str, ...]
    outputs: tuple[str, ...]
    st: np.ndarray
    s1: np.ndarray


def budget_matrix(results: Mapping[str, SobolResult]) -> BudgetMatrix:
    outputs = tuple(results)
    first = results[outputs[0]]
    st = np.column_stack([results[o].st for o in outputs])
    s1 = np.column_stack([results[o].s1 for o in outputs])
    order = np.argsort(-st.max(axis=1))
    return BudgetMatrix(
        factors=tuple(first.factors[i] for i in order),
        outputs=outputs,
        st=st[order],
        s1=s1[order],
    )


def matrix_markdown(matrix: BudgetMatrix, min_share: float = 0.005) -> str:
    """Total-order shares, factors x outputs; rows where every share is < ``min_share`` are
    dropped (their number is stated)."""
    keep = matrix.st.max(axis=1) >= min_share
    head = "| Factor | " + " | ".join(matrix.outputs) + " |\n"
    head += "|---|" + "---:|" * len(matrix.outputs) + "\n"
    rows = [
        f"| {f} | " + " | ".join(f"{v:.3f}" if v >= min_share else "·" for v in row) + " |"
        for f, row, k in zip(matrix.factors, matrix.st, keep, strict=True)
        if k
    ]
    dropped = int((~keep).sum())
    foot = f"\n\n`·` = below {min_share:.1%}." + (
        f" {dropped} further factor(s) contribute less than that to every output."
        if dropped
        else ""
    )
    return head + "\n".join(rows) + foot
