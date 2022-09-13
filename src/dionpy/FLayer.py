from __future__ import annotations

from datetime import datetime
from typing import Tuple, List

import numpy as np
import pymap3d as pm

from .IonLayer import IonLayer
from .modules.helpers import Ellipsoid, check_elaz_shape
from .modules.ion_tools import srange, refr_index, refr_angle, trop_refr, plasfreq


class FLayer(IonLayer):
    """
    Implements a model of ionospheric refraction.

    :param dt: Date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param hbot: Lower limit in [km] of the F layer of the ionosphere.
    :param htop: Upper limit in [km] of the F layer of the ionosphere.
    :param nlayers: Number of sub-layers in the F layer for intermediate calculations.
    :param nside: Resolution of healpix grid.
    :param pbar: If True - a progress bar will appear.
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
        self,
        dt: datetime,
        position: Tuple[float, float, float],
        hbot: float = 150,
        htop: float = 500,
        nlayers: int = 30,
        nside: int = 128,
        pbar: bool = True,
        _autocalc: bool = True,
    ):
        super().__init__(
            dt,
            position,
            hbot,
            htop,
            nlayers,
            nside,
            rdeg=24,
            pbar=pbar,
            _autocalc=_autocalc,
            name="F layer",
        )

    def refr(
        self,
        el: float | np.ndarray,
        az: float | np.ndarray,
        freq: float | List | np.ndarray,
        troposphere: bool = True,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param freq: Frequency of observation(s) in [MHz]. If  - the calculation will be performed in parallel on all
                     available cores. Requires `dt` to be a single datetime object.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :return: Refraction angle in [deg] at given sky coordinates, time and frequency of observation.
        """
        freq *= 1e6
        check_elaz_shape(el, az)
        el, az = el.copy(), az.copy()
        re = 6378100.0
        ell = Ellipsoid()
        f_heights = np.linspace(self.hbot, self.htop, self.nlayers) * 1e3
        delta_theta = 0 * el
        inf_theta_mask = 0 * el
        nan_theta_mask = 0 * el

        theta = np.deg2rad(90 - el)
        if troposphere:
            dtheta = trop_refr(theta)
            theta += dtheta
            el -= np.rad2deg(dtheta)

        # Distance from telescope to first layer
        r_slant = srange(np.deg2rad(90 - el), f_heights[0] - self.position[2])
        # Geodetic coordinates of 'hit point' on the first layer
        lat_ray, lon_ray, _ = pm.aer2geodetic(
            az, el, r_slant, *self.position, ell=ell
        )  # arrays
        # The sides of the 1st triangle
        d_tel = re + self.position[2]  # Distance from Earth center to telescope
        d_cur = re + f_heights[0]  # Distance from Earth center to layer

        # The inclination angle at the 1st interface using law of cosines [rad]
        costheta_inc = (r_slant**2 + d_cur**2 - d_tel**2) / (2 * r_slant * d_cur)
        assert (costheta_inc <= 1).all(), "Something is wrong with coordinates."
        theta_inc = np.arccos(costheta_inc)

        # Refraction index of air
        n_cur = np.ones(el.shape)

        # Get IRI info of point
        fed = self.ed(el, az, layer=0)

        # Refraction index of 1st point
        n_next = refr_index(fed, freq)
        nan_theta_mask += plasfreq(fed) > freq
        # The outgoing angle at the 1st interface using Snell's law
        theta_ref = refr_angle(n_cur, n_next, theta_inc)
        inf_theta_mask += np.abs((n_cur / n_next * np.sin(theta_inc))) > 1


        delta_theta += theta_ref - theta_inc

        el_cur = np.rad2deg(np.pi / 2 - theta_ref)
        n_cur = n_next

        for i in range(1, self.nlayers):
            h_next = f_heights[i]
            d_next = re + h_next

            # Angle between d_cur and r_slant
            int_angle = np.pi - theta_ref
            # The inclination angle at the i-th interface using law of sines [rad]
            theta_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

            # Getting r2 using law of cosines
            r_slant = srange(np.deg2rad(90 - el_cur), d_next - d_cur, re=re + d_cur)
            # Get geodetic coordinates of point
            lat_ray, lon_ray, _ = pm.aer2geodetic(
                az, el_cur, r_slant, lat_ray, lon_ray, f_heights[i - 1], ell=ell
            )
            if i == self.nlayers-1:
                n_next = 1
            else:
                # Get IRI info of 2nd point
                fed = self.ed(el, az, layer=i)

                # Refractive indices
                n_next = refr_index(fed, freq)
                nan_theta_mask += plasfreq(fed) > freq


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
        return np.rad2deg(delta_theta)
