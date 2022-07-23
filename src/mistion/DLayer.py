import itertools
from multiprocessing import cpu_count, Pool

from tqdm import tqdm

import numpy as np
import healpy as hp
import iricore

from mistion.modules.collision_models import col_aggarwal, col_nicolet, col_setty
from mistion.modules.helpers import check_elaz_shape, eval_layer, iri_star
from mistion.modules.ion_tools import d_atten, trop_refr, nu_p


class DLayer:
    def __init__(
            self,
            dt,
            position,
            dbot=60,
            dtop=90,
            ndlayers=10,
            nside=128,
            pbar: bool = True,
            _autocalc: bool = True,
    ):
        self.dbot = dbot
        self.dtop = dtop
        self.ndlayers = ndlayers
        self.dt = dt
        self.position = position

        self.nside = nside
        self._rdeg = 12  # radius of disc queried to healpy
        self._posvec = hp.ang2vec(self.position[1], self.position[0], lonlat=True)
        self._obs_pixels = hp.query_disc(
            self.nside, self._posvec, np.deg2rad(self._rdeg), inclusive=True
        )
        self._obs_lons, self._obs_lats = hp.pix2ang(
            self.nside, self._obs_pixels, lonlat=True
        )
        self.d_e_density = np.zeros((len(self._obs_pixels), ndlayers))
        self.d_e_temp = np.zeros((len(self._obs_pixels), ndlayers))

        if _autocalc:
            self._calc_par(pbar=pbar)

    def _calc(self):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density, electron
        temperature, attenuation factor, average temperature] and F-layer [electron density, angle of the outgoing
        refracted beam at each layer, the net deviation of the elevation angle for each coordinate, refractive index
        at each layer].
        """
        d_heights = np.linspace(self.dbot, self.dtop, self.ndlayers)
        for i in tqdm(range(self.ndlayers)):
            res = iricore.IRI(
                self.dt,
                [d_heights[i], d_heights[i], 1],
                self._obs_lats,
                self._obs_lons,
                replace_missing=0,
            )
            self.d_e_density[:, i] = res["ne"][:, 0]
            self.d_e_temp[:, i] = res["te"][:, 0]
        return

    def _calc_par(self, pbar=True):
        batch = 1000
        nbatches = len(self._obs_pixels) // batch + 1
        nproc = np.min([cpu_count(), nbatches])
        blat = np.array_split(self._obs_lats, nbatches)
        blon = np.array_split(self._obs_lons, nbatches)
        heights = (self.dbot, self.dtop, (self.dtop-self.dbot)/(self.ndlayers-1) - 1e-6)

        with Pool(processes=nproc) as pool:
            res = list(
                tqdm(
                    pool.imap(
                        iri_star,
                        zip(
                            itertools.repeat(self.dt),
                            itertools.repeat(heights),
                            blat,
                            blon,
                            itertools.repeat(0.),
                        ),
                    ),
                    total=nbatches,
                    disable=not pbar,
                    desc="D layer",
                )
            )
            self.d_e_density = np.vstack([r["ne"] for r in res])
            self.d_e_temp = np.vstack([r["te"] for r in res])
        return

    def ded(self, el, az, layer=None):
        return eval_layer(
            el,
            az,
            self.nside,
            self.position,
            self.dbot,
            self.dtop,
            self.ndlayers,
            self._obs_pixels,
            self.d_e_density,
            layer=layer,
        )

    def det(self, el, az, layer=None):
        return eval_layer(
            el,
            az,
            self.nside,
            self.position,
            self.dbot,
            self.dtop,
            self.ndlayers,
            self._obs_pixels,
            self.d_e_temp,
            layer=layer,
        )

    def atten(self, el, az, freq, col_freq="default", troposphere=True):
        """
        Calculates attenuation in D layer for a given model of ionosphere. Output is the attenuation factor between 0
        (total attenuation) and 1 (no attenuation). If coordinates are floats the output will be a single number; if
        they are arrays - the output will be a 2D array with dimensions el.size x az.size.

        Parameters
        ----------
        el : float | np.ndarray
            Elevation of observation(s).
        az : float | np.ndarray
            Azimuth of observation(s).
        freq : float
            Frequency of observations in Hz
        col_freq : str, float
            The collision frequency ('default', 'nicolet', 'setty', 'aggrawal', or float in Hz)
        troposphere : Bool, default=True
            Account for troposphere refraction bias

        Returns
        -------
        np.ndarray
        """
        check_elaz_shape(el, az)
        el, az = el.copy(), az.copy()
        atten = np.empty((*el.shape, self.ndlayers))

        h_d = self.dbot + (self.dtop - self.dbot) / 2
        delta_h_d = self.dtop - self.dbot

        if col_freq == "default" or "aggrawal":
            col_model = col_aggarwal
        elif col_freq == "nicolet":
            col_model = col_nicolet
        elif col_freq == "setty":
            col_model = col_setty
        else:
            col_model = lambda h: np.float64(col_freq)

        heights = np.linspace(self.dbot, self.dtop, self.ndlayers)

        theta = np.deg2rad(90 - el)
        if troposphere:
            dtheta = trop_refr(theta)
            theta += dtheta
            el -= np.rad2deg(dtheta)

        for i in range(self.ndlayers):
            nu_c = col_model(heights[i])
            ded = self.ded(el, az, layer=i)
            plasma_freq = nu_p(ded)
            atten[:, :, i] = d_atten(
                freq, theta, h_d * 1e3, delta_h_d * 1e3, plasma_freq, nu_c
            )
        atten = atten.mean(axis=2)
        if atten.size == 1:
            return atten[0, 0]
        return atten
