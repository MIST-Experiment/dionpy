from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import interp1d


def col_nicolet(h: float | np.ndarray) -> float | np.ndarray:
    """
    Ionosphere collision model from [Nicolet, M. 1953, JATP, 3, 200].

    :param h: Height in km.
    :return: Collision frequency in Hz.
    """
    a = -0.16184565
    b = 28.02068763
    return np.exp(a * h + b)


def col_setty(h: float | np.ndarray) -> float | np.ndarray:
    """
    Ionosphere collision model from [Setty, C. S. G. K. 1972, IJRSP, 1, 38].

    :param h: Height in km.
    :return: Collision frequency in Hz.
    """
    a = -0.16018896
    b = 26.14939429
    return np.exp(a * h + b)


_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_NUC_AGG, _HEI_AGG = np.genfromtxt(
    os.path.join(_CUR_DIR, "col_freq_agg.csv"), delimiter=",", unpack=True
)
_MODEL_AGG = interp1d(_HEI_AGG, _NUC_AGG)


def col_aggarwal(h: float | np.ndarray) -> float | np.ndarray:
    """
    Collision frequency model by (Aggrawal 1979). For details see
    https://ui.adsabs.harvard.edu/abs/1979P%26SS...27..753A/abstract

    :param h: Height in km.
    :return: Collision frequency in Hz.
    """
    return 10 ** _MODEL_AGG(h)
