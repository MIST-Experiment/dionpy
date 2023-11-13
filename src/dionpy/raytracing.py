from typing import Tuple

import numpy as np
import pymap3d as pm

from .IonLayer import IonLayer
from .modules.collision_models import col_aggarwal, col_nicolet, col_setty
from .modules.helpers import Ellipsoid, check_elaz_shape, R_EARTH
from .modules.ion_tools import srange, refr_index, refr_angle, trop_refr, plasfreq

_ROUND_ELL = Ellipsoid(R_EARTH, R_EARTH)
_LIGHT_SPEED = 2.99792458e8  # in [m/s]


def _raytrace_sublayer(lat_ray, lon_ray, h_ray, h_next, alt_cur, az, freq, d_theta, ref_ind, n_sublayer, layer,
                       theta_ref=None):
    # Distance from current position to next layer
    r_slant = srange(np.deg2rad(90 - alt_cur), h_next - h_ray, re=R_EARTH + h_ray)
    lat_next, lon_next, _ = pm.aer2geodetic(az, alt_cur, r_slant, lat_ray, lon_ray, h_ray, ell=_ROUND_ELL)
    # The sides of the 1st triangle
    d_cur = R_EARTH + h_ray  # Distance from Earth center to current point
    d_next = R_EARTH + h_next  # Distance from Earth center to layer

    if theta_ref is None:
        # The inclination angle at the interface using law of cosines [rad]
        costheta_inc = (r_slant ** 2 + d_next ** 2 - d_cur ** 2) / (2 * r_slant * d_next)
        assert (costheta_inc <= 1).all(), (f"Cosine of inclination angle cannot be >= 1. Something is wrong with "
                                           f"coordinates at heights {h_ray * 1e-3:.1f}-{h_next * 1e-3:.1f} [km].")
        theta_inc = np.arccos(costheta_inc)
    else:
        # Angle between d_cur and r_slant
        int_angle_rad = np.pi - theta_ref
        # The inclination angle at the i-th interface using law of sines [rad]
        theta_inc = np.arcsin(np.sin(int_angle_rad) * d_cur / d_next)

    # Get IRI info of point
    ed = layer.edll(lat_next, lon_next, layer=n_sublayer)
    ed = np.where(ed < 0, 0, ed)

    # Refraction index of the surface
    if n_sublayer == layer.nlayers - 1:
        ref_ind_next = np.ones(alt_cur.shape)
        nan_theta_mask = np.zeros(alt_cur.shape)
    else:
        ref_ind_next = refr_index(ed, freq)
        nan_theta_mask = plasfreq(ed, angular=False) > freq

    # The outgoing angle at the 1st interface using Snell's law
    theta_ref = refr_angle(ref_ind, ref_ind_next, theta_inc)
    inf_theta_mask = np.abs((ref_ind / ref_ind_next * np.sin(theta_inc))) > 1
    d_theta += theta_ref - theta_inc
    alt_next = np.rad2deg(np.pi / 2 - theta_ref)
    return lat_next, lon_next, h_next, d_theta, alt_next, ref_ind_next, theta_ref, ed, nan_theta_mask, inf_theta_mask


def raytrace(
        layer_init_dict: dict,
        edens: np.ndarray,
        etemp: np.ndarray,
        alt: float | np.ndarray,
        az: float | np.ndarray,
        freq: float | np.ndarray,
        col_freq: str = "default",
        troposphere: bool = True,
        height_profile: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: fix nans and infs
    # IonLayer initialization with edens and etemp arrays from shared memory
    assert layer_init_dict['autocalc'] is False, "autocalc param should be False, check IonFrame."

    layer = IonLayer(**layer_init_dict)
    layer.edens = edens
    layer.etemp = etemp

    # Initialization of variables
    # - General
    freq *= 1e6
    check_elaz_shape(alt, az)
    alt_cur = np.array(alt)
    az = np.array(az)
    heights = layer.get_heights() * 1e3  # in [m]

    # - For refraction
    delta_theta = 0 * alt_cur
    delta_theta_hist = np.empty((*alt_cur.shape, layer.nlayers))
    inf_theta_mask = 0 * alt_cur
    nan_theta_mask = 0 * alt_cur

    # - For absorption and emission
    dh = (layer.htop - layer.hbot) / layer.nlayers * 1e3  # in [m]
    atten = np.empty((*alt_cur.shape, layer.nlayers))
    emiss = np.empty((*alt_cur.shape, layer.nlayers))

    if troposphere:
        alt_cur -= trop_refr(alt_cur, layer.position[-1])

    col_freq_choices = {
        "default": col_aggarwal,
        "aggrawal": col_aggarwal,
        "nicolet": col_nicolet,
        "setty": col_setty,
    }

    try:
        col_model = col_freq_choices[col_freq]
    except KeyError:
        if isinstance(col_freq, float):
            col_model = lambda h: np.float64(col_freq)
        else:
            raise ValueError(f"The col_freq parameter must be one of {list(col_freq_choices.keys())} or a float im Hz.")

    # Init values for the first sub-layer
    ref_ind_cur = np.ones(alt_cur.shape)
    lat_ray, lon_ray, h_ray = layer.position
    theta_ref = None

    for i in range(layer.nlayers):
        # Calculating absorption and emission: part 1
        freq_c = col_model(heights[i] * 1e-3)
        et = layer.et(alt_cur, az, layer=i)
        freq_om = freq * 2 * np.pi
        ds = (srange(np.deg2rad(90 - alt_cur), heights[i] + 0.5 * dh) -
              srange(np.deg2rad(90 - alt_cur), heights[i] - 0.5 * dh))

        # Tracing change in position due to refraction
        lat_ray, lon_ray, h_ray, delta_theta, alt_cur, ref_ind_cur, theta_ref, ed, nt_mask, it_mask = _raytrace_sublayer(
            lat_ray, lon_ray, h_ray, heights[i], alt_cur, az, freq, delta_theta, ref_ind_cur, i, layer, theta_ref)
        delta_theta_hist[..., i] = delta_theta
        inf_theta_mask += it_mask
        nan_theta_mask += nt_mask

        # Calculating absorption and emission: part 2
        freq_p = plasfreq(ed)
        atten[..., i] = np.exp(-0.5 * freq_p ** 2 / (freq_om ** 2 + freq_c ** 2) * freq_c * ds / _LIGHT_SPEED)
        emiss[..., i] = (1 - atten[..., i]) * et

    delta_theta = np.where(inf_theta_mask == 0, delta_theta, np.inf)
    delta_theta = np.where(nan_theta_mask == 0, delta_theta, np.nan)

    if not height_profile:
        atten = atten.prod(axis=-1)
        emiss = emiss.sum(axis=-1)
        return np.rad2deg(delta_theta), atten, emiss
    else:
        return np.rad2deg(delta_theta_hist), atten, emiss


def raytrace_star(args):
    """
    For parallel calculations
    """
    return raytrace(*args)
