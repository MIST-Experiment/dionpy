from __future__ import annotations

from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, Union, List, Sequence

import numpy as np
from numpy import ndarray

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
    :param iriversion: Version of the IRI model to use. Must be a two digit integer that refers to
                        the last two digits of the IRI version number. For example, version 20 refers
                        to IRI-2020.
    :param echaim: Use ECHAIM model for electron density estimation.
    :param autocalc: If True - the model will be calculated immediately after definition.
    :param pbar: If True - a progress bar will appear.
    """

    def __init__(
            self,
            dt: datetime,
            position: Sequence[float, float, float],
            hbot: float = 60,
            htop: float = 90,
            nlayers: int = 100,
            nside: int = 64,
            iriversion: int = 20,
            echaim: bool = False,
            autocalc: bool = True,
            pbar: bool = True,
            _pool: Union[Pool, None] = None,
            **kwargs
    ):
        super().__init__(
            dt,
            position,
            hbot,
            htop,
            nlayers,
            nside,
            pbar=pbar,
            name="D layer",
            iriversion=iriversion,
            echaim=echaim,
            autocalc=autocalc,
            _pool=_pool,
            **kwargs
        )

    def atten(
            self,
            alt: float | np.ndarray,
            az: float | np.ndarray,
            freq: float | np.ndarray,
            col_freq: str = "default",
            emission: bool = False,
            troposphere: bool = True,
            height_profile: bool = False,
    ) -> ndarray | Tuple[ndarray, ndarray]:
        """
        :param alt: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param freq: Frequency of observation(s) in [MHz]. If array - the calculation will be performed in parallel on
                     all available cores. Requires `dt` to be a single datetime object.
        :param col_freq: Collision frequency model. Available options: 'default', 'nicolet', 'setty', 'aggrawal',
                         or float in Hz.
        :param emission: If True - also returns array of emission temperatures.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param height_profile: If True - returns height profile of attenuation (+1 dimension).
        :return: Attenuation factor at given sky coordinates, time and frequency of observation. Output is the
                 attenuation factor between 0 (total attenuation) and 1 (no attenuation).
        """
        freq_om = freq * 1e6 * 2 * np.pi    # Converting to angular (omega) frequency and going MHz -> Hz
        check_elaz_shape(alt, az)
        alt = np.asarray(alt)
        az = np.asarray(az)
        alt, az = alt.copy(), az.copy()
        atten = np.empty((*alt.shape, self.nlayers))
        emiss = np.empty((*alt.shape, self.nlayers))
        dh = (self.htop - self.hbot) / self.nlayers * 1e3   # in [m]

        col_model = col_aggarwal

        heights_km = np.linspace(self.hbot, self.htop, self.nlayers)

        theta = np.deg2rad(90 - alt)
        if troposphere:
            dtheta = trop_refr(alt, self.position[-1] * 1e-3)
            theta += np.deg2rad(dtheta)
            alt -= dtheta

        c = 2.99792458e8    # [m/s]

        for i in range(self.nlayers):
            freq_c = col_model(heights_km[i])
            ded = self.ed(alt, az, layer=i)
            det = self.et(alt, az, layer=i)
            freq_p = plasfreq(ded)
            ds = srange(theta, heights_km[i] * 1e3 + 0.5 * dh) - srange(theta, heights_km[i] * 1e3 - 0.5 * dh)
            atten[..., i] = np.exp(-0.5 * freq_p**2 / (freq_om**2 + freq_c**2) * freq_c * ds / c)
            emiss[..., i] = (1 - atten[..., i]) * det

        if not height_profile:
            atten = atten.prod(axis=-1)
            emiss = emiss.sum(axis=-1)

        return atten, emiss
