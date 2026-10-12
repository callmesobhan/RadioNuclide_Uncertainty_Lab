"""MIRD formalism, ``D = A~ x S``, in the package's unit system (MBq, h, kg, Gy)."""

from __future__ import annotations

import numpy as np

MEV_TO_J = 1.602176634e-13  # exact (2019 SI)
DECAYS_PER_MBQ_H = 1.0e6 * 3600.0  # 1 MBq h = 3.6e9 decays


def self_s_value(
    energy_per_decay_mev: float | np.ndarray,
    absorbed_fraction: float | np.ndarray,
    mass_kg: float | np.ndarray,
):
    """Self-irradiation S-value in Gy / (MBq h) for a uniformly filled source organ.

    ``S = E * phi / m`` where ``E`` is the mean *non-penetrating* energy emitted per decay
    (electrons: beta, conversion, Auger), ``phi`` the fraction of it absorbed in the organ
    and ``m`` the organ mass. Photon cross-dose and penetrating-radiation self-dose are not
    included: use tabulated S-values (e.g. phantom-based) for those contributions.
    """
    return energy_per_decay_mev * MEV_TO_J * absorbed_fraction / mass_kg * DECAYS_PER_MBQ_H


def absorbed_dose_gy(tia_mbq_h: float | np.ndarray, s_value_gy_per_mbq_h: float | np.ndarray):
    """``D = A~ x S`` in Gy, with ``A~`` in MBq h and ``S`` in Gy / (MBq h)."""
    return tia_mbq_h * s_value_gy_per_mbq_h


def tia_coefficients_from_doses(
    s_matrix_gy_per_mbq_h: np.ndarray, doses_gy_per_gbq: np.ndarray
) -> np.ndarray:
    """Time-integrated-activity coefficients (h) that reproduce given organ doses.

    ``S T = D`` with ``S[t, s] = S(t <- s)`` in Gy/(MBq h), ``T[s]`` the coefficient of source ``s``
    in hours (TIA per administered MBq) and ``D[t]`` the *total* dose per administered activity.
    Gy/GBq and mGy/MBq are the same number, so published doses can be passed as printed. Solving
    the whole system, instead of dividing each dose by its self S-value, keeps self and cross-organ
    contributions consistent: the model then returns exactly these doses at nominal inputs.
    """
    s = np.asarray(s_matrix_gy_per_mbq_h, dtype=float)
    d = np.asarray(doses_gy_per_gbq, dtype=float) / 1000.0  # Gy/GBq -> Gy/MBq
    if s.ndim != 2 or s.shape[0] != s.shape[1] or d.shape != (s.shape[0],):
        raise ValueError("need a square S matrix and one dose per target")
    t = np.linalg.solve(s, d)
    if np.any(t <= 0):
        raise ValueError(
            f"doses are inconsistent with the S matrix: non-positive TIA coefficients {t}"
        )
    return t
