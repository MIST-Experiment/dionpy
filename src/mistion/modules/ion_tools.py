import numpy as np

def srange(theta, alt, R_E=6378100.0):
    """
    Calculates the distance in meters from the telescope to the point (theta, alt).

    Parameters
    ----------
    theta : float | np.ndarray
        Zenith angle in radians
    alt : float
        Altitude in meters
    R_E : float, optional
        Radius of the Earth in meters

    Returns
    -------
    r : float
        Range in meters
    """
    r = -R_E * np.cos(theta) + np.sqrt(
        (R_E * np.cos(theta)) ** 2 + alt**2 + 2 * alt * R_E
    )
    return r


def nu_p(n_e):
    """
    Plasma frequency of cold electrons

    Parameters
    ----------
    n_e : float | np.ndarray
        Electron density

    Returns
    -------
    float
        Plasma frequency in Hz
    """
    e = 1.60217662e-19
    m_e = 9.10938356e-31
    epsilon0 = 8.85418782e-12
    if np.min(n_e) < 0:
        raise ValueError(
            "Number density cannot be < 0. Most probably iri2016 does not include data for the specified date."
        )
    return 1 / (2 * np.pi) * np.sqrt((n_e * e**2) / (m_e * epsilon0))


def n_f(n_e, freq):
    """
    Refractive index of F-layer from electron density

    Parameters
    ----------
    n_e : float | np.ndarray
        Electron density
    freq : float
        Signal frequency in Hz
    """
    return (1 - (nu_p(n_e) / freq) ** 2) ** 0.5


def refr_angle(n1, n2, phi):
    """
    Angle of refracted ray using Snell's law.

    Parameters
    ----------
    n1 : float | np.ndarray
        Refractive index in previous medium
    n2 : float | np.ndarray
        Refractive index in current medium
    phi : float | np.ndarray
        Angle of incident ray in rad

    Returns
    -------
    float
        Angle in rad
    """
    return np.arcsin(n1 / n2 * np.sin(phi))


def trop_refr(theta):
    """
    Calculates the tropospheric refraction (delta theta).

    Parameters
    ----------
    theta : float | array_like
        Zenith angle in radians

    Returns
    -------
    dtheta : float | array_like
        Change of the angle theta due to tropospheric refraction (in radians).

    Notes
    -----
    Approximation is recommended by the ITU-R:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-7-201510-S!!PDF-E.pdf
    """
    a = 16709.51
    b = -19066.21
    c = 5396.33
    return 1 / (a + b * theta + c * theta**2)


def d_atten(nu, theta, h_d, delta_hd, nu_p, nu_c):
    """
    Calculates the attenuation factor from frequency of the signal [Hz], angle [rad],
    altitude of the D-layer midpoint [km], thickness of the D-layer [km], plasma frequency [Hz],
    and electron collision frequency [Hz]. Output is the attenuation factor between 0 (total attenuation)
    and 1 (no attenuation).
    """
    R_E = 6378100
    c = 2.99792458e8
    delta_s = (
        delta_hd * (1 + h_d / R_E) * (np.cos(theta) ** 2 + 2 * h_d / R_E) ** (-0.5)
    )
    f = np.exp(-(2 * np.pi * nu_p**2 * nu_c * delta_s) / (c * (nu_c**2 + nu**2)))
    return f
