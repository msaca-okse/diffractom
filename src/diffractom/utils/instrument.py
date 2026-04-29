"""Instrument-specific functions for neutron Bragg-edge tomography.

Each function in this module is tied to a particular instrument and should be
passed explicitly to :func:`~diffractom.operators.bragg_edge_tomographic_operator.build_bragg_matrix_cpu`
(and the corresponding operator class) via the ``pulse_tail_fn`` argument.
This keeps all beamline-specific empirical parameters out of the physics code.

Instrument functions
--------------------
``pulse_tail_fn(lam) -> np.ndarray``
    Maps wavelength λ (Å) to the pulse-tail length parameter τ (units chosen
    so that ``α = τ / 10000`` gives α in Å in the peak-shape formula of
    ``xs_singlecrystal_2022.m``).  See:

        A1(:,12) = (tau(A1(:,5)) ./ 10000);  % alfa

Currently provided
------------------
:func:`raden_pulse_tail`
    RADEN beamline, J-PARC (Tokai, Japan).
"""

from __future__ import annotations

import numpy as np


def raden_pulse_tail(lam: np.ndarray) -> np.ndarray:
    """Empirical pulse-tail parameter τ(λ) for the RADEN beamline at J-PARC.

    Direct translation of ``tau.m`` supplied with the MATLAB matrix-generation
    code (``Matrix_gen.m``) used at RADEN/J-PARC.

    The fitted polynomial reads::

        p1 = 1.39341;  p2 = 0.18492
        p3 = 18.94806; p4 = -10.82914;  p5 = 16.6964
        tau = erf((λ - p1) / p2) * (p3 + p4*λ) + p5*λ

    In the peak-shape calculation (``xs_singlecrystal_2022.m``), τ is used as::

        α = τ(λ₀) / 10000          # pulse-tail length in Å

    Parameters
    ----------
    lam : np.ndarray
        Wavelength grid in Å.

    Returns
    -------
    np.ndarray
        τ(λ) values (same shape as *lam*).  Divide by 10000 to obtain α in Å.

    References
    ----------
    Implemented from ``tau.m`` (F. Malamud, J-PARC / RADEN group).
    Instrument: RADEN, BL22, Materials and Life Science Experimental Facility
    (MLF), J-PARC, Tokai, Japan.
    """
    from scipy.special import erf as sp_erf

    lam = np.asarray(lam, dtype=np.float64)
    p1, p2 = 1.39341, 0.18492
    p3, p4, p5 = 18.94806, -10.82914, 16.6964
    return sp_erf((lam - p1) / p2) * (p3 + p4 * lam) + p5 * lam
