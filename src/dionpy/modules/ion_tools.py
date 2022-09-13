from __future__ import annotations

import numpy as np

np.seterr(invalid="ignore")


def srange(
    theta: float | np.ndarray, alt: float, re: float = 6378100
) -> float | np.ndarray:
    """
    :param theta: Zenith angle in [rad].
    :param alt: Altitude in [m].
    :param re: Radius of the Earth in [m].
    :return: Distance in meters from the telescope to the point (theta, alt)
    """
    r = -re * np.cos(theta) + np.sqrt(
        (re * np.cos(theta)) ** 2 + alt**2 + 2 * alt * re
    )
    return r


def plasfreq(n_e: float | np.ndarray) -> float | np.ndarray:
    """
    :param n_e: Electron density in [m^-3].
    :return: Plasma frequency of cold electrons in Hz.
    """
    e = 1.60217662e-19
    m_e = 9.10938356e-31
    epsilon0 = 8.85418782e-12
    if np.min(n_e) < 0:
        raise ValueError(
            "Number density cannot be < 0. Most probably iricore does not include data for the specified date. Please "
            "update the library by calling iricore.update()."
        )
    return 1 / (2 * np.pi) * np.sqrt((n_e * e**2) / (m_e * epsilon0))


def refr_index(n_e: float | np.ndarray, freq: float):
    """

    :param n_e: Electron density in [m^-3].
    :param freq: Observational frequency in [Hz].
    :return: Refractive index of the ionosphere from electron density.
    """
    nu_p = plasfreq(n_e)
    return np.sqrt(1 - (nu_p / freq) ** 2)



def refr_angle(
    n1: float | np.ndarray,
    n2: float | np.ndarray,
    phi: float | np.ndarray,
) -> float | np.ndarray:
    """
    Snell's law.

    :param n1: Refractive index in previous medium.
    :param n2: Refractive index in current medium.
    :param phi: Angle of incident ray in [rad].
    :return: Outcoming angle in [rad].
    """
    return np.arcsin(n1 / n2 * np.sin(phi))


def trop_refr(theta: float | np.ndarray) -> float | np.ndarray:
    """
    :param theta: Zenith angle in radians.
    :return: Change of the angle theta due to tropospheric refraction (in radians).
    """
    a = 16709.51
    b = -19066.21
    c = 5396.33
    return 1 / (a + b * theta + c * theta**2)

#
# def _d_atten_low(freq, theta, h_d, delta_hd, freq_p, freq_c):
#     re = 6378100
#     c = 2.99792458e8
#     delta_s = (
#         delta_hd * (1 + h_d / re) * (np.cos(theta) ** 2 + 2 * h_d / re) ** (-0.5)
#     )
#     datten = np.exp(
#         -(2 * np.pi * freq_p ** 2 * freq_c * delta_s) / (c * (freq_c ** 2 + freq ** 2))
#     )
#     return datten
#
#
# def _d_atten_high(freq, theta, h_d, delta_hd, freq_p, freq_c):
#     itheta = np.deg2rad(np.linspace(70, 85, 50))
#     iatten = _d_atten_low(freq, itheta, h_d, delta_hd, freq_p, freq_c)
#     deg = 2
#     pol = np.poly1d(np.polyfit(itheta, iatten, deg))
#     return pol(theta)
#
#
# def d_atten(
#     freq: float,
#     theta: float | np.ndarray,
#     h_d: float,
#     delta_hd: float,
#     freq_p: float,
#     freq_c: float,
# ) -> float | np.ndarray:
#     """
#
#     :param freq: Frequensy of observation in [Hz].
#     :param theta: Zenith angle in [rad].
#     :param h_d: Altitude of the D-layer midpoint in [km].
#     :param delta_hd: Thickness of the D-layer in [km].
#     :param freq_p: Plasma frequency in [Hz].
#     :param freq_c: Electron collision frequency in [Hz].
#     :return: Attenuation factor between 0 (total attenuation) and 1 (no attenuation).
#     """
#     if theta < 80:
#         return _d_atten_low(freq, theta, h_d, delta_hd, freq_p, freq_c)
#     return _d_atten_high(freq, theta, h_d, delta_hd, freq_p, freq_c)
#
#
# d_atten = np.vectorize(d_atten)
