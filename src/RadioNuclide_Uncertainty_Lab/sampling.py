"""Unit-hypercube sample generators.

All randomness in the package flows through ``unit_samples`` so a single integer seed makes a
whole analysis reproducible.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

METHODS = ("sobol", "lhs", "random")


def unit_samples(n: int, d: int, method: str = "sobol", seed: int | None = None) -> np.ndarray:
    """Return an ``(n, d)`` array in the open unit hypercube.

    ``sobol`` is a scrambled Sobol' sequence; its balance properties need a power-of-two
    length, so ``n`` is rounded **up** to the next power of two (check ``len(result)``).
    ``lhs`` is a Latin hypercube and ``random`` plain pseudo-random sampling; both return
    exactly ``n`` rows.
    """
    if n < 2 or d < 1:
        raise ValueError(f"need n >= 2 and d >= 1, got n={n}, d={d}")
    if method == "sobol":
        m = math.ceil(math.log2(n))
        return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
    if method == "lhs":
        return qmc.LatinHypercube(d, seed=seed).random(n)
    if method == "random":
        return np.random.default_rng(seed).random((n, d))
    raise ValueError(f"unknown sampling method {method!r}; choose from {METHODS}")
