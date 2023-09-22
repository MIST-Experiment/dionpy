from typing import List

import numpy as np
import pymap3d as pm

from .IonLayer import IonLayer
from .modules.helpers import Ellipsoid, check_elaz_shape, R_EARTH
from .modules.ion_tools import srange, refr_index, refr_angle, trop_refr, plasfreq

# TODO: create a separate sublayer refraction calculation function for F-layer

def raytrace(
        layer_init_dict: dict,
        edens: np.ndarray,
        etemp: np.ndarray,
        alt: float | np.ndarray,
        az: float | np.ndarray,
        freq: float | np.ndarray,
        col_freq: str = "default",
        troposphere: bool = True,
        height_profile: bool = False
) -> List[float | np.ndarray]:
    # TODO: fix nans and infs
    # IonLayer initialization with edens and etemp arrays from shared memory
    assert layer_init_dict['autocalc'] is False, "autocalc param should be False, check IonFrame."
    layer = IonLayer(**layer_init_dict)
    layer.edens = edens
    layer.etemp = etemp

    # Initialization of variables
    freq *= 1e6
    check_elaz_shape(alt, az)
    alt = np.array(alt)
    az = np.array(az)
    ell = Ellipsoid(R_EARTH, R_EARTH)
    heights = layer.get_heights() * 1e3  # in [m]

    delta_theta = 0 * alt
    inf_theta_mask = 0 * alt
    nan_theta_mask = 0 * alt

    if troposphere:
        alt -= trop_refr(alt, layer.position[-1])

    # Calculations for the first sub-layer

    # Distance from telescope to first layer
    r_slant = srange(np.deg2rad(90 - alt), heights[0] - layer.position[-1])
    # Geodetic coordinates of 'hit point' on the first layer
    lat_ray, lon_ray, _ = pm.aer2geodetic(
        az, alt, r_slant, *layer.position, ell=ell
    )  # arrays
    # The sides of the 1st triangle
    d_tel = R_EARTH + layer.position[2]  # Distance from Earth center to telescope
    d_cur = R_EARTH + heights[0]  # Distance from Earth center to layer

    # The inclination angle at the 1st interface using law of cosines [rad]
    costheta_inc = (r_slant ** 2 + d_cur ** 2 - d_tel ** 2) / (2 * r_slant * d_cur)
    assert (costheta_inc <= 1).all(), "Something is wrong with coordinates."
    theta_inc = np.arccos(costheta_inc)

    # Refraction index of air
    n_cur = np.ones(alt.shape)

    # Get IRI info of point
    fed = layer.edll(lat_ray, lon_ray, layer=0)
    fed = np.where(fed < 0, 0, fed)
    # Refraction index of 1st point
    n_next = refr_index(fed, freq)
    nan_theta_mask += plasfreq(fed, angular=False) > freq
    # The outgoing angle at the 1st interface using Snell's law
    theta_ref = refr_angle(n_cur, n_next, theta_inc)
    inf_theta_mask += np.abs((n_cur / n_next * np.sin(theta_inc))) > 1

    delta_theta += theta_ref - theta_inc

    el_cur = np.rad2deg(np.pi / 2 - theta_ref)
    n_cur = n_next

    for i in range(1, layer.nlayers):
        h_next = heights[i]
        d_next = R_EARTH + h_next

        # Angle between d_cur and r_slant
        int_angle = np.pi - theta_ref
        # The inclination angle at the i-th interface using law of sines [rad]
        theta_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

        # Getting r2 using law of cosines
        r_slant = srange(np.deg2rad(90 - el_cur), d_next - d_cur, re=R_EARTH + d_cur)
        # Get geodetic coordinates of point
        lat_ray, lon_ray, _ = pm.aer2geodetic(
            az, el_cur, r_slant, lat_ray, lon_ray, heights[i - 1], ell=ell
        )
        if i == layer.nlayers - 1:
            n_next = 1
        else:
            # Get IRI info of 2nd point
            fed = layer.edll(lat_ray, lon_ray, layer=i)
            fed = np.where(fed < 0, 0, fed)
            # Refractive indices
            n_next = refr_index(fed, freq)
            nan_theta_mask += plasfreq(fed, angular=False) > freq

        # The outgoing angle at the 2nd interface using Snell's law
        theta_ref = refr_angle(n_cur, n_next, theta_inc)
        inf_theta_mask += np.abs((n_cur / n_next * np.sin(theta_inc))) > 1
        delta_theta += theta_ref - theta_inc

        # Update variables for new interface
        el_cur = np.rad2deg(np.pi / 2 - theta_ref)
        n_cur = n_next
        d_cur = d_next

    delta_theta = np.where(inf_theta_mask == 0, delta_theta, np.inf)
    delta_theta = np.where(nan_theta_mask == 0, delta_theta, np.nan)

    return delta_theta