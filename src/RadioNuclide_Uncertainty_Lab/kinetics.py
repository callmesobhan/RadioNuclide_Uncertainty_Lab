"""Biokinetics: time-activity curves and time-integrated activity (TIA).

Units used throughout the package: activity in MBq, time in hours, rates in 1/h.
Rates passed to the functions below are **effective** (biological + physical decay) unless a
function says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .distributions import lognormal, truncated_normal
from .inputs import Parameter

LN2 = math.log(2.0)


def decay_constant(half_life_h: float | np.ndarray) -> float | np.ndarray:
    return LN2 / half_life_h


def effective_rate(
    physical_half_life_h: float | np.ndarray, biological_half_life_h: float | np.ndarray
) -> float | np.ndarray:
    """Effective clearance rate ``ln2/Tp + ln2/Tb`` (1/h)."""
    return LN2 / physical_half_life_h + LN2 / biological_half_life_h


def mono_exponential_tia(a0_mbq: float | np.ndarray, rate_per_h: float | np.ndarray):
    """Analytic ``integral_0^inf a0 * exp(-rate t) dt`` (MBq h). Instant-uptake model."""
    return a0_mbq / rate_per_h


def multi_exponential_tia(amplitudes_mbq: np.ndarray, rates_per_h: np.ndarray) -> np.ndarray:
    """``sum_k A_k / lambda_k`` over the last axis. A negative amplitude models an uptake phase."""
    return np.sum(np.asarray(amplitudes_mbq) / np.asarray(rates_per_h), axis=-1)


def time_activity(t_h: np.ndarray, amplitudes_mbq: np.ndarray, rates_per_h: np.ndarray):
    t = np.asarray(t_h, dtype=float)[..., None]
    return np.sum(np.asarray(amplitudes_mbq) * np.exp(-np.asarray(rates_per_h) * t), axis=-1)


def trapezoid_tia(t_h: np.ndarray, activity_mbq: np.ndarray, physical_half_life_h: float) -> float:
    """Trapezoidal integral over the measured points plus a physical-decay-only tail.

    This is the model-free baseline: no curve assumptions between time points, and the tail
    assumes no further biological clearance (conservative). Time points must be ascending;
    activity at t=0 is taken as 0 unless a t=0 point is given.
    """
    t = np.asarray(t_h, dtype=float)
    a = np.asarray(activity_mbq, dtype=float)
    if t.shape != a.shape or t.ndim != 1 or len(t) < 2:
        raise ValueError("t_h and activity_mbq must be 1-D, same length, at least 2 points")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time points must be strictly ascending")
    if t[0] > 0:
        t, a = np.concatenate([[0.0], t]), np.concatenate([[0.0], a])
    body = float(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(t)))
    tail = float(a[-1] / decay_constant(physical_half_life_h))
    return body + tail


@dataclass(frozen=True)
class MonoExpFit:
    """Result of fitting ``A(t) = a0 * exp(-rate * t)``; ``cov`` is for ``(a0, rate)``."""

    a0_mbq: float
    rate_per_h: float
    cov: np.ndarray
    n_points: int

    @property
    def tia_mbq_h(self) -> float:
        return self.a0_mbq / self.rate_per_h

    @property
    def effective_half_life_h(self) -> float:
        return LN2 / self.rate_per_h

    def to_parameters(
        self,
        group: str = "biokinetic_fit",
        prefix: str = "fit_",
        marginal: str = "lognormal",
    ) -> tuple[list[Parameter], dict[tuple[str, str], float]]:
        """Fit uncertainty as ready-to-use inputs: two correlated parameters in one group.

        ``marginal='lognormal'`` (default) matches each parameter's mean and SD with a
        log-normal, which cannot go negative and has the right skew when the relative
        uncertainty is large (the rate of a poorly determined fit easily reaches 30 %; a normal
        then puts mass near zero and ``a0 / rate`` develops absurd tails). The latent
        correlation is chosen so the **Pearson covariance of the fit is reproduced exactly**.
        ``marginal='truncated_normal'`` keeps normal marginals truncated at zero and uses the
        fit correlation unchanged; adequate only while relative uncertainties stay below ~10 %.
        """
        if marginal not in ("lognormal", "truncated_normal"):
            raise ValueError("marginal must be 'lognormal' or 'truncated_normal'")
        mean = np.array([self.a0_mbq, self.rate_per_h])
        sd = np.sqrt(np.diag(self.cov))
        rho = float(self.cov[0, 1] / (sd[0] * sd[1]))
        names = (f"{prefix}a0_mbq", f"{prefix}rate_per_h")
        units = ("MBq", "1/h")
        source = f"mono-exponential fit, {self.n_points} points; mean/covariance from the fit"

        if marginal == "truncated_normal":
            dists = [truncated_normal(m, s, low=0.0) for m, s in zip(mean, sd, strict=True)]
            latent = rho
        else:
            cv = sd / mean
            sigma = np.sqrt(np.log1p(cv**2))  # log-space SD of each parameter
            dists = [
                lognormal(m / math.sqrt(1 + c**2), math.exp(s))  # median = mean / sqrt(1 + cv^2)
                for m, c, s in zip(mean, cv, sigma, strict=True)
            ]
            # cov(X, Y) = E[X] E[Y] (exp(rho_z s1 s2) - 1)  =>  solve for the latent rho_z
            latent = float(np.log1p(rho * cv[0] * cv[1]) / (sigma[0] * sigma[1]))
            if not -1 < latent < 1:
                raise ValueError(
                    f"fit covariance is not representable with log-normal marginals "
                    f"(latent correlation {latent:.3f})"
                )
        params = [
            Parameter(n, d, unit=u, group=group, source=source)
            for n, d, u in zip(names, dists, units, strict=True)
        ]
        return params, {(names[0], names[1]): latent}


def fit_mono_exponential(
    t_h: np.ndarray, activity_mbq: np.ndarray, sigma_mbq: np.ndarray | None = None
) -> MonoExpFit:
    """Weighted non-linear least squares for ``a0 * exp(-rate * t)``.

    With ``sigma_mbq`` (1-sigma measurement uncertainty per point) the covariance is the
    absolute one; without it, scaled by the residual variance.
    """
    t = np.asarray(t_h, dtype=float)
    a = np.asarray(activity_mbq, dtype=float)
    if len(t) < 3:
        raise ValueError("need at least 3 points to estimate a covariance")
    if np.any(a <= 0):
        raise ValueError("activities must be > 0 for the log-linear starting guess")
    slope, intercept = np.polyfit(t, np.log(a), 1)
    popt, pcov = curve_fit(
        lambda x, a0, k: a0 * np.exp(-k * x),
        t,
        a,
        p0=[math.exp(intercept), max(-slope, 1e-6)],
        sigma=sigma_mbq,
        absolute_sigma=sigma_mbq is not None,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
    )
    return MonoExpFit(float(popt[0]), float(popt[1]), pcov, len(t))
