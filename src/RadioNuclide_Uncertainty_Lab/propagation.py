"""Monte Carlo propagation of input uncertainty to the model outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .model import UncertainModel
from .sampling import unit_samples


@dataclass(frozen=True)
class Summary:
    """Distributional summary of one output.

    ``mean_se`` is the iid Monte Carlo standard error ``sd / sqrt(n)``. For quasi-random
    (Sobol') and Latin-hypercube samples this is conservative: the true error is smaller.
    The interval is the central percentile interval, so it is valid for skewed outputs.
    """

    output: str
    unit: str
    n: int
    mean: float
    sd: float
    mean_se: float
    median: float
    level: float
    ci_low: float
    ci_high: float
    skewness: float
    gsd: float | None  # geometric SD; None unless every sample is > 0

    @property
    def cv(self) -> float:
        return self.sd / abs(self.mean) if self.mean else math.inf

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {**self.__dict__, "cv": self.cv}


@dataclass(frozen=True)
class UncertaintyResult:
    model_name: str
    output_names: tuple[str, ...]
    output_units: tuple[str, ...]
    inputs: dict[str, np.ndarray]
    values: np.ndarray  # shape (n, number of outputs)
    method: str
    seed: int | None

    @property
    def n(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, output: str) -> np.ndarray:
        return self.values[:, self.output_names.index(output)]

    @property
    def output(self) -> np.ndarray:
        """The output array of a single-output model."""
        if len(self.output_names) != 1:
            raise ValueError("several outputs; index the result by name, e.g. result['kidneys']")
        return self.values[:, 0]

    def _resolve(self, output: str | None) -> str:
        if output is None:
            if len(self.output_names) != 1:
                raise ValueError(f"several outputs {list(self.output_names)}; pass output=...")
            return self.output_names[0]
        if output not in self.output_names:
            raise KeyError(f"unknown output {output!r}; choose from {list(self.output_names)}")
        return output

    def summary(self, output: str | None = None, level: float = 0.95) -> Summary:
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        name = self._resolve(output)
        y = self[name]
        lo, hi = np.quantile(y, [(1 - level) / 2, (1 + level) / 2])
        sd = float(np.std(y, ddof=1))
        gsd = float(math.exp(np.std(np.log(y), ddof=1))) if np.all(y > 0) else None
        return Summary(
            output=name,
            unit=self.output_units[self.output_names.index(name)],
            n=len(y),
            mean=float(np.mean(y)),
            sd=sd,
            mean_se=sd / math.sqrt(len(y)),
            median=float(np.median(y)),
            level=level,
            ci_low=float(lo),
            ci_high=float(hi),
            skewness=float(stats.skew(y)),
            gsd=gsd,
        )

    def summaries(self, level: float = 0.95) -> dict[str, Summary]:
        return {name: self.summary(name, level) for name in self.output_names}

    def correlation(self, a: str, b: str) -> float:
        """Pearson correlation between two outputs (induced by shared inputs)."""
        return float(np.corrcoef(self[a], self[b])[0, 1])


def propagate(
    model: UncertainModel,
    n: int = 2**14,
    method: str = "sobol",
    seed: int | None = None,
) -> UncertaintyResult:
    """Draw ``n`` input samples (rounded up to a power of two for ``method='sobol'``),
    evaluate the model and return the samples with the outputs."""
    u = unit_samples(n, model.inputs.dim, method=method, seed=seed)
    x, y = model.evaluate(u)
    return UncertaintyResult(
        model_name=model.name,
        output_names=tuple(model.outputs),
        output_units=tuple(model.unit(o) for o in model.outputs),
        inputs=x,
        values=y,
        method=method,
        seed=seed,
    )
