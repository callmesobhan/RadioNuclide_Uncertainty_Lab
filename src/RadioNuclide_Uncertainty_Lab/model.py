"""The object everything else operates on: an uncertain input space plus a dose function."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .inputs import InputSpace

ModelFunc = Callable[[Mapping[str, np.ndarray]], "np.ndarray | Mapping[str, np.ndarray]"]


@dataclass
class UncertainModel:
    """``func`` maps ``{parameter name: array of shape (n,)}`` to the model outputs.

    A single-output model returns an array ``(n,)``. A multi-output model (e.g. one dose per
    target organ, plus derived quantities such as dose ratios) returns either a mapping
    ``{output name: (n,)}`` or an array ``(n, m)`` in the order of ``outputs``. All outputs come
    from the *same* input samples, so their joint behaviour (shared systematic errors, ratios)
    is preserved.

    ``func`` must be vectorised (no Python loops over samples) and side-effect free; it is called
    many times by propagation and by the Sobol' estimator. Constants belong in the closure, not in
    the input space: only quantities with an uncertainty are inputs.
    """

    name: str
    inputs: InputSpace
    func: ModelFunc
    outputs: str | Sequence[str] = "absorbed_dose"
    output_unit: str = "Gy"  # default unit of every output ...
    output_units: Mapping[str, str] = field(default_factory=dict)  # ... unless overridden here
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outputs = (self.outputs,) if isinstance(self.outputs, str) else tuple(self.outputs)
        if not self.outputs:
            raise ValueError("a model needs at least one output")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"duplicate output names: {self.outputs}")

    def unit(self, output: str) -> str:
        return self.output_units.get(output, self.output_unit)

    def resolve_output(self, output: str | None) -> str:
        """Name of the requested output; ``None`` is allowed only for single-output models."""
        if output is None:
            if len(self.outputs) != 1:
                raise ValueError(f"model has several outputs {list(self.outputs)}; pass output=...")
            return self.outputs[0]
        if output not in self.outputs:
            raise KeyError(f"unknown output {output!r}; choose from {list(self.outputs)}")
        return output

    def evaluate(self, u: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Evaluate on unit-cube samples; returns ``(physical inputs, outputs of shape (n, m))``."""
        x = self.inputs.transform(u)
        raw = self.func(x)
        n, m = len(u), len(self.outputs)
        if isinstance(raw, Mapping):
            missing = [k for k in self.outputs if k not in raw]
            if missing:
                raise ValueError(f"model did not return outputs {missing}")
            y = np.column_stack(
                [np.broadcast_to(np.asarray(raw[k], float), (n,)) for k in self.outputs]
            )
        else:
            y = np.asarray(raw, dtype=float)
            if y.ndim == 1:
                y = y[:, None]
        if y.shape != (n, m):
            raise ValueError(f"model returned shape {y.shape}, expected ({n}, {m})")
        bad = ~np.isfinite(y)
        if bad.any():
            row, col = (int(i[0]) for i in np.nonzero(bad))
            offending = {k: float(v[row]) for k, v in x.items()}
            raise ValueError(
                f"model returned {int(bad.sum())} non-finite value(s); first in output "
                f"{self.outputs[col]!r} at sample {row} with inputs {offending}"
            )
        return x, y
