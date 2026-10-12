"""Run-time diagnostics: warn when an analysis is likely to mislead.

Every rule below corresponds to a way this kind of analysis goes wrong silently, several of them
found while reproducing published results (see docs/benchmarks.md). Diagnostics never change a
number; they travel with the report so a reader can judge it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RunSettings
from .model import UncertainModel
from .propagation import Summary, UncertaintyResult
from .sensitivity import RankSensitivity, SobolResult

WARNING, INFO = "warning", "info"


@dataclass(frozen=True)
class Diagnostic:
    level: str
    code: str
    message: str


def diagnose(
    model: UncertainModel,
    run: RunSettings,
    result: UncertaintyResult,
    summaries: dict[str, Summary],
    ranks: dict[str, RankSensitivity],
    sobol: dict[str, SobolResult] | None,
) -> tuple[Diagnostic, ...]:
    out: list[Diagnostic] = []
    add = lambda level, code, msg: out.append(Diagnostic(level, code, msg))  # noqa: E731

    if run.seed is None:
        add(WARNING, "NO_SEED", "no seed set: the result cannot be reproduced bit-for-bit")
    if result.n < 2**12:
        add(WARNING, "SMALL_SAMPLE", f"only {result.n} Monte Carlo samples; use at least 4096")

    for p in model.inputs.parameters:
        mean, sd = float(p.dist.mean()), float(p.dist.std())
        if sd == 0:
            add(WARNING, "ZERO_VARIANCE_INPUT", f"{p.name} has no uncertainty; make it a constant")
        elif mean > 0 and float(p.dist.ppf(1e-4)) <= 0:
            add(
                WARNING,
                "NON_POSITIVE_SUPPORT",
                f"{p.name}: relative SD {sd / mean:.0%} lets a positive quantity reach zero "
                "(1e-4 quantile <= 0). Dividing by it or raising it to a negative power produces "
                "absurd tails; use a lognormal.",
            )

    for o, s in summaries.items():
        if abs(s.skewness) > 3 or s.cv > 1:
            add(
                WARNING,
                "HEAVY_TAIL",
                f"{o}: skewness {s.skewness:.1f}, CV {s.cv:.0%}; check the "
                "marginals of the inputs that enter non-linearly",
            )
        if s.mean and s.mean_se / abs(s.mean) > 0.01:
            add(
                WARNING,
                "MC_ERROR",
                f"{o}: Monte Carlo error of the mean is {s.mean_se / abs(s.mean):.1%}; increase n",
            )
        r = ranks[o]
        if r.r2 < 0.7:
            add(
                INFO,
                "NON_LINEAR",
                f"{o}: linear R² = {r.r2:.2f}; SRC is not a variance share, "
                "rely on the Sobol' budget",
            )

    if sobol is None:
        add(INFO, "NO_SOBOL", "variance-based budget not computed")
    else:
        if next(iter(sobol.values())).n < 2**11:
            add(WARNING, "SMALL_SOBOL", "Sobol' base sample below 2048; indices will be noisy")
        for o, sob in sobol.items():
            half = (sob.st_ci[:, 1] - sob.st_ci[:, 0]) / 2
            wide = [
                f
                for f, st, h in zip(sob.factors, sob.st, half, strict=True)
                if st >= 0.05 and h > 0.05
            ]
            if wide:
                add(
                    WARNING,
                    "WIDE_SOBOL_CI",
                    f"{o}: total-order CI half-width > 0.05 for {wide}; increase sobol_n",
                )
            if sob.s1.sum() > 1.05:
                add(
                    WARNING,
                    "SOBOL_INCONSISTENT",
                    f"{o}: sum of first-order indices is "
                    f"{sob.s1.sum():.2f} > 1; the estimate is unreliable (increase sobol_n)",
                )
            bad = [
                f for f, s1, st in zip(sob.factors, sob.s1, sob.st, strict=True) if st < s1 - 0.05
            ]
            if bad:
                add(WARNING, "SOBOL_ORDER", f"{o}: total-order below first-order for {bad}")
            neg = [f for f, s1 in zip(sob.factors, sob.s1, strict=True) if s1 < -0.05]
            if neg:
                add(WARNING, "NEGATIVE_INDEX", f"{o}: clearly negative first-order index for {neg}")
            if sob.interaction_share > 0.2:
                add(
                    INFO,
                    "INTERACTIONS",
                    f"{o}: {sob.interaction_share:.0%} of the variance is "
                    "interaction; single-factor attributions understate each factor's role",
                )
    # stable order: warnings first, then info
    out.sort(key=lambda d: (d.level != WARNING, d.code))
    return tuple(out)
