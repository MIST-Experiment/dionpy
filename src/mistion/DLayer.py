import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools

import numpy as np
import pymap3d as pm
import iricore

from mistion.modules.collision_models import col_aggarwal, col_nicolet, col_setty
from mistion.modules.helpers import check_latlon, Ellipsoid, generate_grid, OrderError
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
        elrange=None,
        azrange=None,
        gridsize=50,
    ):
        check_latlon(position[0], position[1])
        self.dbot = dbot
        self.dtop = dtop
        self.ndlayers = ndlayers
        self.dt = dt
        self.lat0, self.lon0, self.alt0 = position

        if elrange is None:
            self.elrange = (0, 90)
        else:
            self.elrange = elrange

        if azrange is None:
            self.azrange = (0, 360)
        else:
            self.azrange = azrange

        self.gridsize = gridsize

        self._d_e_density = None
        self._d_e_temp = None

        self._interp_ded = None
        self._interp_deda = None
        self._interp_det = None
        self._interp_deta = None

    def _interpolate_d_layer(self, kind="cubic"):
        from scipy.interpolate import interp2d

        az_vals = np.linspace(*self.azrange, self.gridsize, endpoint=True)
        el_vals = np.linspace(*self.elrange, self.gridsize)
        self._interp_ded = [
            interp2d(
                el_vals,
                az_vals,
                self._d_e_density[:, i].reshape(self.gridsize, self.gridsize),
                kind=kind,
            )
            for i in range(self.ndlayers)
        ]
        aver_data = self._d_e_density.mean(axis=1)
        self._interp_deda = interp2d(
            el_vals,
            az_vals,
            aver_data.reshape(self.gridsize, self.gridsize),
            kind=kind,
        )
        self._interp_det = [
            interp2d(
                el_vals,
                az_vals,
                self._d_e_temp[:, i].reshape(self.gridsize, self.gridsize),
                kind=kind,
            )
            for i in range(self.ndlayers)
        ]
        self._interp_deta = interp2d(
            el_vals,
            az_vals,
            self._d_e_temp.mean(axis=1).reshape(self.gridsize, self.gridsize),
            kind=kind,
        )

    def calc(self, nproc=1, pbar=True, batch=500):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density, electron
        temperature, attenuation factor, average temperature] and F-layer [electron density, angle of the outgoing
        refracted beam at each layer, the net deviation of the elevation angle for each coordinate, refractive index
        at each layer].
        """
        el, az = generate_grid(*self.elrange, *self.azrange, self.gridsize)
        cpus = cpu_count()
        if cpus < nproc:
            nproc = cpus
            warnings.warn(
                f"You have only {cpus} cpu threads available. Setting number of processes to {cpus}.",
                RuntimeWarning,
                stacklevel=2,
            )
        nbatches = len(el) // batch + 1
        if nbatches < nproc:
            nbatches = len(el) // 150 + 1
            warnings.warn(
                f"Selected batch size is not optimal. Setting batch size to 150.",
                RuntimeWarning,
                stacklevel=2,
            )
        el_batches = np.array_split(el, nbatches)
        az_batches = np.array_split(az, nbatches)

        with Pool(processes=nproc) as pool:
            dlayer = list(
                tqdm(
                    pool.imap(
                        _d_temp_density_star,
                        zip(
                            itertools.repeat(self.dt),
                            itertools.repeat(self.dbot),
                            itertools.repeat(self.dtop),
                            itertools.repeat(self.ndlayers),
                            el_batches,
                            az_batches,
                            itertools.repeat(self.lat0),
                            itertools.repeat(self.lon0),
                            itertools.repeat(self.alt0),
                        ),
                    ),
                    total=len(el_batches),
                    disable=not pbar,
                    desc="D-layer",
                )
            )
            self._d_e_density = np.vstack([d[0] for d in dlayer])
            self._d_e_temp = np.vstack([d[1] for d in dlayer])

        self._interpolate_d_layer()
        return

    def _check_elaz(self, el, az, size_err=True):
        if el is None or az is None:
            el = np.linspace(*self.elrange, self.gridsize)
            az = np.linspace(*self.azrange, self.gridsize)
        else:
            el = np.asarray(el)
            az = np.asarray(az)
            if el.size != az.size and size_err:
                raise ValueError("Elevation and azimuth must have the same size")
        return el, az

    def datten(self, freq, el=None, az=None, col_freq="default", troposhpere=True):
        """
        Calculates attenuation in D layer for a given model of ionosphere. Output is the attenuation factor between 0
        (total attenuation) and 1 (no attenuation). If coordinates are floats the output will be a single number; if
        they are arrays - the output will be a 2D array with dimensions el.size x az.size.

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.
        col_freq : str, float
            The collision frequency ('default', 'nicolet', 'setty', 'aggrawal', or float in Hz)
        troposhpere : Bool, default=True
            Account for troposphere refraction bias

        Returns
        -------
        np.ndarray
        """
        if self._interp_ded is None:
            raise OrderError(
                "You must calculate the model first. Try running [model].calc()"
            )
        el, az = self._check_elaz(el, az)
        h_d = self.dbot + (self.dtop - self.dbot) / 2
        delta_h_d = self.dtop - self.dbot
        d_attenuation = np.empty((el.size, az.size, self.ndlayers))

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
        if troposhpere:
            theta += trop_refr(theta)

        for i in range(self.ndlayers):
            nu_c = col_model(heights[i])
            ne = self._interp_ded[i](el, az)
            ne = np.where(ne > 0, ne, 0)
            plasma_freq = nu_p(ne)
            d_attenuation[:, :, i] = d_atten(
                freq, theta, h_d, delta_h_d, plasma_freq, nu_c
            )
        atten = d_attenuation.mean(axis=2)
        if atten.size == 1:
            return atten[0, 0]
        return atten

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
