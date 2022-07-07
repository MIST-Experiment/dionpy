import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools

import numpy as np
import pymap3d as pm
import healpy as hp
import iricore

from mistion.modules.collision_models import col_aggarwal, col_nicolet, col_setty
from mistion.modules.helpers import Ellipsoid, sky2ll, check_elaz_shape
from mistion.modules.ion_tools import srange, d_atten, trop_refr, nu_p


def _d_temp_density(dt, d_bot, d_top, ndlayers, els, azs, lat, lon, alt):
    """
    # TODO
    """
    assert len(els) == len(azs), "Elevations and azimuths must have the same size."
    npoints = len(els)
    d_heights = np.linspace(d_bot, d_top, ndlayers)
    d_srange = np.empty((npoints, ndlayers))
    ell = Ellipsoid()
    for i in range(ndlayers):
        d_srange[:, i] = srange((90 - els) * np.pi / 180, d_heights[i] * 1e3)

    d_obs_lat = np.empty((npoints, ndlayers))
    d_obs_lon = np.empty((npoints, ndlayers))
    d_obs_h = np.empty((npoints, ndlayers))

    for i in range(ndlayers):
        d_obs_lat[:, i], d_obs_lon[:, i], d_obs_h[:, i] = pm.aer2geodetic(
            azs, els, d_srange[:, i], lat, lon, alt, ell=ell
        )

    d_e_density = np.empty((npoints, ndlayers))
    d_e_temp = np.empty((npoints, ndlayers))

    for i in range(ndlayers):
        res = iricore.IRI(
            dt,
            [d_heights[i], d_heights[i], 1],
            d_obs_lat[:, i],
            d_obs_lon[:, i],
            replace_missing=0,
        )
        d_e_density[:, i] = res["ne"][:, 0]
        d_e_temp[:, i] = res["te"][:, 0]

    return d_e_density, d_e_temp


def _d_temp_density_star(pars):
    return _d_temp_density(*pars)


class DLayer:
    def __init__(
            self,
            dt,
            position,
            dbot=60,
            dtop=90,
            ndlayers=10,
            nside=256,
    ):
        self.dbot = dbot
        self.dtop = dtop
        self.ndlayers = ndlayers
        self.dt = dt
        self.position = position

        self.nside = nside
        self._rdeg = 15  # radius of disc queried to healpy
        self._posvec = hp.ang2vec(self.position[1], self.position[0], lonlat=True)
        self._obs_pixels = hp.query_disc(
            self.nside, self._posvec, np.deg2rad(self._rdeg), inclusive=True
        )
        self._obs_lons, self._obs_lats = hp.pix2ang(
            self.nside, self._obs_pixels, lonlat=True
        )

        self._d_e_density = np.empty((len(self._obs_pixels), ndlayers))
        self._d_e_temp = np.empty((len(self._obs_pixels), ndlayers))

        self._calc()

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
            self._d_e_density[:, i] = res["ne"][:, 0]
            self._d_e_temp[:, i] = res["te"][:, 0]
        return

    def ded(self, el, az, layer=None):
        check_elaz_shape(el, az)
        dheights = np.linspace(self.dbot, self.dtop, self.ndlayers)
        map_ = np.zeros(hp.nside2npix(self.nside)) + hp.UNSEEN
        if layer is None:
            ded = np.empty((*el.shape, self.ndlayers))
            for i in range(self.ndlayers):
                map_[self._obs_pixels] = self._d_e_density[:, i]
                obs_lat, obs_lon = sky2ll(el, az, dheights[i], self.position)
                # print(np.min(obs_lat), np.max(obs_lat))
                # print(np.min(obs_lon), np.max(obs_lon))
                ded[:, :, i] = hp.pixelfunc.get_interp_val(
                    map_, obs_lon, obs_lat, lonlat=True
                )
            return ded.mean(axis=2)
        elif isinstance(layer, int) and layer < self.ndlayers + 1:
            map_[self._obs_pixels] = self._d_e_density[:, layer]
            obs_lat, obs_lon = sky2ll(el, az, dheights[layer], self.position)
            ded = hp.pixelfunc.get_interp_val(
                map_, obs_lon, obs_lat, lonlat=True
            )
            return ded
        else:
            raise ValueError(f"The layer value must be integer and be in range [0, {self.ndlayers-1}]")

    def datten(self, el, az, freq, col_freq="default", troposphere=True):
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
        datten = np.empty((*el.shape, self.ndlayers))

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
            print(np.rad2deg(trop_refr(theta)))
            theta += trop_refr(theta)
            el -= np.rad2deg(trop_refr(theta))
        for i in range(self.ndlayers):
            nu_c = col_model(heights[i])
            ded = self.ded(el, az, layer=i)
            plasma_freq = nu_p(ded)
            datten[:, :, i] = d_atten(freq, theta, h_d, delta_h_d, plasma_freq, nu_c)
        datten = datten.mean(axis=2)
        if datten.size == 1:
            return datten[0, 0]
        return datten

    def datten_rough(self, freq, el=None, az=None, troposphere=True):
        """
        Function for rough estimation of the attenuation in the D layer if IRI data is not available. See appendix A of
        (Monsalve et al., 2021 ApJ 908 145) for details.

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.
        troposphere : Bool, default=True
            Account for troposphere refraction bias.

        Returns
        -------
        np.ndarray
        """
        el, az = self._check_elaz(el, az)

        theta = np.deg2rad(90 - el)
        if troposphere:
            theta += trop_refr(theta)

        h_d = 75.0
        delta_h_d = 30.0
        plasma_freq = np.ones((el.size, az.size)) * nu_p(1e8)
        nu_c = 5e6
        atten = d_atten(freq, theta, h_d, delta_h_d, plasma_freq, nu_c)
        if atten.size == 1:
            return atten[0, 0]
        return atten
