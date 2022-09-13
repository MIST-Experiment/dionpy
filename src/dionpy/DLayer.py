from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np

from .IonLayer import IonLayer
from .modules.collision_models import col_aggarwal, col_nicolet, col_setty
from .modules.helpers import check_elaz_shape
from .modules.ion_tools import trop_refr, plasfreq, srange


class DLayer(IonLayer):
    """
    Implements a model of ionospheric attenuation.

    :param dt: Date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param hbot: Lower limit in [km] of the D layer of the ionosphere.
    :param htop: Upper limit in [km] of the D layer of the ionosphere.
    :param nlayers: Number of sub-layers in the D layer for intermediate calculations.
    :param nside: Resolution of healpix grid.
    :param pbar: If True - a progress bar will appear.
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt: datetime,
            position: Tuple[float, float, float],
            hbot: float = 60,
            htop: float = 90,
            nlayers: int = 10,
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
            rdeg=12,
            pbar=pbar,
            _autocalc=_autocalc,
            name="D layer",
        )

    def atten(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            freq: float | np.ndarray,
            col_freq: str = "default",
            troposphere: bool = True,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param freq: Frequency of observation(s) in [MHz]. If  - the calculation will be performed in parallel on all
                     available cores. Requires `dt` to be a single datetime object.
        :param col_freq: Collision frequency model. Available options: 'default', 'nicolet', 'setty', 'aggrawal',
                         or float in Hz.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :return: Attenuation factor at given sky coordinates, time and frequency of observation. Output is the
                 attenuation factor between 0 (total attenuation) and 1 (no attenuation).
        """
        freq *= 1e6
        check_elaz_shape(el, az)
        el, az = el.copy(), az.copy()
        atten = np.empty((*el.shape, self.nlayers))
        dh = (self.htop - self.hbot) / self.nlayers * 1e3

        if col_freq == "default" or "aggrawal":
            col_model = col_aggarwal
        elif col_freq == "nicolet":
            col_model = col_nicolet
        elif col_freq == "setty":
            col_model = col_setty
        else:
            col_model = lambda h: np.float64(col_freq)

        heights_km = np.linspace(self.hbot, self.htop, self.nlayers)

        theta = np.deg2rad(90 - el)
        if troposphere:
            dtheta = trop_refr(theta)
            theta += dtheta
            el -= np.rad2deg(dtheta)

        c = 2.99792458e8  # speed of light

        for i in range(self.nlayers):
            freq_c = col_model(heights_km[i])
            ded = self.ed(el, az, layer=i)
            freq_p = plasfreq(ded)
            ds = srange(theta, heights_km[i] * 1e3 + 0.5 * dh) - srange(theta, heights_km[i] * 1e3 - 0.5 * dh)
            atten[:, :, i] = np.exp(-2 * np.pi * freq_p ** 2 * freq_c * ds / (freq ** 2 + freq_c ** 2) / c)
        # atten = 1 + atten.sum(axis=2) - self.nlayers
        atten = atten.prod(axis=2)

        if atten.size == 1:
            return atten[0, 0]
        return atten
