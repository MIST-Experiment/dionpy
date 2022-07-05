import os
import tempfile
import itertools as it
import warnings
from multiprocessing import Pool, cpu_count
from datetime import datetime
from time import time

import numpy as np
import iricore
import pymap3d as pm
from tqdm import tqdm
from scipy.interpolate import interp1d

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()


class OrderError(Exception):
    """
    Exception indicating incorrect order of simulation routines.
    """

    pass


class Ellipsoid:
    """
    Custom ellipsoid for pymap3d package. Implements a simple sphere.
    """

    def __init__(self):
        self.semimajor_axis = 6378100.0
        self.semiminor_axis = 6378100.0
        self.flattening = 0.0
        self.thirdflattening = 0.0
        self.eccentricity = 0.0


def _none_or_array(vals):
    if vals is None:
        return None
    return np.array(vals)


def check_latlon(lat, lon):
    if not -90 <= lat <= 90:
        raise ValueError("Latitude of the instrument must be in range [-90, 90]")
    if not -180 <= lon < 180:
        raise ValueError("Longitude of the instrument must be in range [-180, 180]")


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


def col_nicolet(height):
    """
    #TODO
    """
    a = -0.16184565
    b = 28.02068763
    return np.exp(a * height + b)


def col_setty(height):
    """
    #TODO
    """
    a = -0.16018896
    b = 26.14939429
    return np.exp(a * height + b)

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_NUC_AGG, _HEI_AGG = np.genfromtxt(os.path.join(_CUR_DIR, "col_freq_agg.csv"), delimiter=",", unpack=True)
_MODEL_AGG = interp1d(_HEI_AGG, _NUC_AGG)


def col_aggarwal(h):
    """
    Collision frequency model by Aggrawal 1979. For details see
    https://ui.adsabs.harvard.edu/abs/1979P%26SS...27..753A/abstract

    Parameters
    ----------
    h : float | np.ndarray
        Altitude in km

    Returns
    -------
    nuc : float | np.ndarray
        Electron collision frequency
    """
    return 10 ** _MODEL_AGG(h)


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


def nu_p_warm(n_e, T_e, freq):
    """
    Plasma frequency of warm electrons

    Parameters
    ----------
    n_e : float | np.ndarray
        Electron density
    T_e : float
        Electron temperature
    freq : float
        Frequency of observation

    Returns
    -------
    float
        Plasma frequency in Hz
    """
    k_B = 1.38064852e-23
    e = 1.60217662e-19
    m_e = 9.10938356e-31
    epsilon0 = 8.85418782e-12
    c = 299792458

    if np.min(n_e) < 0:
        raise ValueError(
            "Number density cannot be < 0. Most probably iri2016 does not include data for the specified date."
        )
    if np.min(T_e) < 0:
        raise ValueError(
            "Temperaturey cannot be < 0. Most probably iri2016 does not include data for the specified date."
        )

    k = c / freq
    om2_p = (n_e * e**2) / (m_e * epsilon0)
    om2_t = 3 * k**2 * k_B * T_e / m_e
    return 0.5 * np.sqrt(om2_p + om2_t) / np.pi


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

    # return d_e_density, d_e_temp, d_obs_lat, d_obs_lon
    return d_e_density, d_e_temp


def _d_temp_density_star(pars):
    return _d_temp_density(*pars)


def _calc_flayer(dt, f_bot, f_top, nflayers, els, azs, lat, lon, alt, freq):
    """
    #TODO
    """
    R_E = 6378100.0
    ell = Ellipsoid()
    ncoords = len(els)
    f_heights = np.linspace(f_bot, f_top, nflayers) * 1000
    ns = np.empty((ncoords, nflayers))  # refractive indices
    f_e_density = np.empty((ncoords, nflayers))
    phis = np.empty((ncoords, nflayers))  # angles of refraction
    delta_theta = np.zeros(ncoords)  # total change in angl(ncoords, nflayers)e

    # Distance from telescope to first layer
    r_slant = srange((90.0 - els) * np.pi / 180.0, f_heights[0] - alt)
    # Geodetic coordinates of 'hit point' on the first layer
    lat_ray, lon_ray, _ = pm.aer2geodetic(
        azs, els, r_slant, lat, lon, alt, ell=ell
    )  # arrays
    # The sides of the 1st triangle
    d_tel = R_E + alt  # Distance from Earth center to telescope
    d_cur = R_E + f_heights[0]  # Distance from Earth center to layer

    # The inclination angle at the 1st interface using law of cosines [rad]
    costheta_inc = (r_slant**2 + d_cur**2 - d_tel**2) / (2 * r_slant * d_cur)
    assert (costheta_inc <= 1).all(), "Something is wrong with coordinates."
    theta_inc = np.arccos(costheta_inc)

    # Refraction index of air
    n_cur = np.ones(ncoords)

    # Get IRI info of point
    f_alt_prof = iricore.IRI(
        dt, [f_bot * 1e-3, f_bot * 1e-3, 1], lat_ray, lon_ray, replace_missing=0.0
    )
    f_e_density[:, 0] = f_alt_prof["ne"][:, 0]

    # Refraction index of 1st point
    n_next = n_f(f_e_density[:, 0], freq)
    ns[:, 0] = n_next[:]

    # The outgoing angle at the 1st interface using Snell's law
    theta_ref = refr_angle(n_cur, n_next, theta_inc)
    phis[:, 0] = theta_ref
    delta_theta += theta_ref - theta_inc
    el_cur = np.rad2deg(np.pi / 2 - theta_ref)
    n_cur = n_next

    for i in range(1, nflayers):
        h_next = f_heights[i]
        d_next = R_E + h_next

        # Angle between d_cur and r_slant
        int_angle = np.pi - theta_ref
        # The inclination angle at the i-th interface using law of sines [rad]
        theta_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

        # Getting r2 using law of cosines
        r_slant = srange(
            (90.0 - el_cur) * np.pi / 180.0, d_next - d_cur, R_E=R_E + d_cur
        )
        # Get geodetic coordinates of point
        lat_ray, lon_ray, _ = pm.aer2geodetic(
            azs, el_cur, r_slant, lat_ray, lon_ray, f_heights[i - 1], ell=ell
        )
        # Get IRI info of 2nd point
        f_alt_prof = iricore.IRI(
            dt,
            [f_heights[i] * 1e-3, f_heights[i] * 1e-3, 1],
            lat_ray,
            lon_ray,
            replace_missing=0,
        )
        f_e_density[:, i] = f_alt_prof["ne"][:, 0]

        # Refractive indices
        n_next = n_f(f_e_density[:, i], freq)
        ns[:, i] = n_next

        # If this is the last point then use refractive index of vacuum
        if i == nflayers - 1:
            n_next = np.ones(ncoords)

        # The outgoing angle at the 2nd interface using Snell's law
        theta_ref = refr_angle(n_cur, n_next, theta_inc)
        phis[:, i] = theta_ref
        delta_theta += theta_ref - theta_inc

        # Update variables for new interface
        el_cur = np.rad2deg(np.pi / 2 - theta_ref)
        n_cur = n_next
        d_cur = d_next

    return f_e_density, phis, delta_theta, ns


def _calc_flayer_star(pars):
    return _calc_flayer(*pars)


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
    return 1 / (a + b * theta + c * theta * theta)


class IonModel:
    """
    Parameters
    ----------
    lat0 : float
        Latitude of the instrument in degrees
    lon0 : float
        Longitude of the instrument in degrees
    alt0 : float
        Altitude of the instrument in meters
    freq : float
        Frequency in Hz of signal at which all model values will be calculated
    dt : datetime
        Date and time of observation in format "yyyy-mm-dd hh:mm"
    """

    def __init__(self, lat0, lon0, alt0, freq, dt):
        check_latlon(lat0, lon0)

        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0
        self.freq = freq
        self.dt = dt

        self._npoints = None
        self._gridsize = None
        self.el = None
        self.az = None

        self.ndlayers = None
        self.d_bot = None
        self.d_top = None

        self._d_e_density = None
        self._d_e_temp = None
        self._d_avg_temp = None

        self.nflayers = None
        self.f_bot = None
        self.f_top = None

        self._f_e_density = None
        self._phis = None
        self._delta_theta = None
        self._ns = None

        self._interp_d_layers = None
        self._interp_d_aver = None
        self._interp_d_temp = None
        self._interp_f_layers = None
        self._interp_f_aver = None
        self._interp_f_temp = None
        self._interp_delta_theta = None


    def _set_coords(self, el, az, gridsize):
        self.az = az
        self.el = el
        self._npoints = len(el)
        self._gridsize = gridsize


    def generate_grid(
        self, el_start=0.0, el_end=90.0, az_start=0.0, az_end=360.0, gridsize=64
    ):
        """
        Generates a grid of coordinates at which all the parameters will be calculated.

        Parameters
        ----------
        el_start : float, optional
            The starting value of the sequence of elevations (in degrees).
        el_end : float, optional
            The end value of the sequence of elevations (in degrees).
        az_start : float, optional
            The starting value of the sequence of azimuths (in degrees).
        az_end : float, optional
            The end value of the sequence of azimuths (in degrees).
        gridsize : int, optional
            Resolution of the coordinate grid. The total number of points will be [gridsize x gridsize].
        """

        az_vals = np.linspace(az_start, az_end, gridsize, endpoint=True)
        el_vals = np.linspace(el_start, el_end, gridsize)
        # Alternative to az, el = np.meshgrid(az, el), then flatten
        self.az = np.repeat(az_vals, gridsize)
        self.el = np.tile(el_vals, gridsize)
        self._gridsize = gridsize
        self._npoints = gridsize * gridsize

    def set_lprops(
        self, ndlayers=10, d_bot=60, d_top=90, nflayers=40, f_bot=150, f_top=500
    ):
        """
        Set up all necessary parameters for the ionosphere model

        Parameters
        ----------
        ndlayers : int
            Number of layers in D-layer
        d_bot : float
            Lower limit of the D-layer in km
        d_top : float
            Upper limit of the D-layer in km
        nflayers : int
            Number of layers in F-layer
        f_bot : float
            Lower limit of the F-layer in km
        f_top : float
            Upper limit of the F-layer in km
        """
        self.ndlayers = ndlayers
        self.d_bot = d_bot
        self.d_top = d_top
        self.nflayers = nflayers
        self.f_bot = f_bot
        self.f_top = f_top

    def _interpolate_d_layer(self, kind="cubic"):
        from scipy.interpolate import interp2d

        az_vals = np.linspace(
            np.min(self.az), np.max(self.az), self._gridsize, endpoint=True
        )
        el_vals = np.linspace(np.min(self.el), np.max(self.el), self._gridsize)
        lmodels = [
            interp2d(
                el_vals,
                az_vals,
                self._d_e_density[:, i].reshape(self._gridsize, self._gridsize),
                kind=kind,
            )
            for i in range(self.ndlayers)
        ]
        self._interp_d_layers = lmodels
        aver_data = self._d_e_density.mean(axis=1)
        self._interp_d_aver = interp2d(
            el_vals,
            az_vals,
            aver_data.reshape(self._gridsize, self._gridsize),
            kind=kind,
        )
        self._interp_d_temp = [
            interp2d(
                el_vals,
                az_vals,
                self._d_e_temp[:, i].reshape(self._gridsize, self._gridsize),
                kind=kind,
            )
            for i in range(self.ndlayers)
        ]
        self._interp_d_avg_temp = interp2d(
            el_vals,
            az_vals,
            self._d_avg_temp.reshape(self._gridsize, self._gridsize),
            kind=kind,
        )

    def _interpolate_f_layer(self, kind="cubic"):
        from scipy.interpolate import interp2d

        az_vals = np.linspace(
            np.min(self.az), np.max(self.az), self._gridsize, endpoint=True
        )
        el_vals = np.linspace(np.min(self.el), np.max(self.el), self._gridsize)
        lmodels = [
            interp2d(
                el_vals,
                az_vals,
                self._f_e_density[:, i].reshape(self._gridsize, self._gridsize),
                kind=kind,
            )
            for i in range(self.nflayers)
        ]
        self._interp_f_layers = lmodels
        aver_data = self._f_e_density.mean(axis=1)
        self._interp_f_aver = interp2d(
            el_vals,
            az_vals,
            aver_data.reshape(self._gridsize, self._gridsize),
            kind=kind,
        )
        self._interp_delta_theta = interp2d(
            el_vals,
            az_vals,
            self._delta_theta.reshape(self._gridsize, self._gridsize),
            kind=kind,
        )

    def calc(self, nproc=1, pbar=True, layer=None, batch=500):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density, electron
        temperature, attenuation factor, average temperature] and F-layer [electron density, angle of the outgoing
        refracted beam at each layer, the net deviation of the elevation angle for each coordinate, refractive index
        at each layer].
        """

        if (
            None in [self.ndlayers, self.d_bot, self.d_top]
            and layer != "f"
            and layer != "F"
        ):
            raise OrderError(
                "You have to set up parameters for the D layer first (use the setup_model() method)"
            )

        if (
            None in [self.nflayers, self.f_bot, self.f_top]
            and layer != "d"
            and layer != "D"
        ):
            raise OrderError(
                "You have to set up parameters for the F layer first (use the setup_model() method)"
            )

        cpus = cpu_count()
        if cpus < nproc:
            nproc = cpus
            warnings.warn(
                f"You have only {cpus} cpu threads available. Setting number of processes to {cpus}.",
                RuntimeWarning,
                stacklevel=2,
            )
        nbatches = self._npoints // batch + 1
        if nbatches < nproc:
            nbatches = self._npoints // 150 + 1
            warnings.warn(
                f"Selected batch size is not optimal. Setting batch size to 150.",
                RuntimeWarning,
                stacklevel=2,
            )
        el_batches = np.array_split(self.el, nbatches)
        az_batches = np.array_split(self.az, nbatches)

        # D layer
        if layer != "f" and layer != "F":
            if not pbar:
                print(
                    "Starting calulation for D layer for date " + str(self.dt),
                    flush=True,
                )
                t1_d = time()
            with Pool(processes=nproc) as pool:
                dlayer = list(
                    tqdm(
                        pool.imap(
                            _d_temp_density_star,
                            zip(
                                it.repeat(self.dt),
                                it.repeat(self.d_bot),
                                it.repeat(self.d_top),
                                it.repeat(self.ndlayers),
                                el_batches,
                                az_batches,
                                it.repeat(self.lat0),
                                it.repeat(self.lon0),
                                it.repeat(self.alt0),
                            ),
                        ),
                        total=len(el_batches),
                        disable=not pbar,
                        desc="D-layer",
                    )
                )
                self._d_e_density = np.vstack([d[0] for d in dlayer])
                self._d_e_temp = np.vstack([d[1] for d in dlayer])
                # self.d_obs_lat = np.vstack([d[2] for d in dlayer])
                # self.d_obs_lon = np.vstack([d[3] for d in dlayer])

            self._d_avg_temp = self._d_e_temp.mean(axis=1)
            self._interpolate_d_layer()
            if not pbar:
                print(
                    f"Calulation for D layer have ended with {time() - t1_d:.1f} seconds.",
                    flush=True,
                )

        # F layer
        if layer != "d" and layer != "D":
            if not pbar:
                print(
                    "Starting calulation for F layer for date " + str(self.dt),
                    flush=True,
                )
                t1_f = time()
            with Pool(processes=nproc) as pool:
                flayer = list(
                    tqdm(
                        pool.imap(
                            _calc_flayer_star,
                            zip(
                                it.repeat(self.dt),
                                it.repeat(self.f_bot),
                                it.repeat(self.f_top),
                                it.repeat(self.nflayers),
                                el_batches,
                                az_batches,
                                it.repeat(self.lat0),
                                it.repeat(self.lon0),
                                it.repeat(self.alt0),
                                it.repeat(self.freq),
                            ),
                        ),
                        total=len(el_batches),
                        disable=not pbar,
                        desc="F-layer",
                    )
                )
                self._f_e_density = np.vstack([f[0] for f in flayer])
                self._phis = np.vstack([f[1] for f in flayer])
                self._delta_theta = np.hstack([f[2] for f in flayer]).reshape([-1])
                self._ns = np.vstack([f[3] for f in flayer])
            self._interpolate_f_layer()
            if not pbar:
                print(
                    f"Calulation for F layer have ended with {time() - t1_f:.1f} seconds.",
                    flush=True,
                )

        return

    def _check_elaz(self, el, az):
        if el is None or az is None:
            if self.el is None:
                raise ValueError(
                    "No coordinates provided. Pass coordinates as arguments or generate a coordinate grid "
                    "with IonModel.generate_grid()."
                )
            el = np.linspace(self.el.min(), self.el.max(), self._gridsize)
            az = np.linspace(self.az.min(), self.az.max(), self._gridsize)
        else:
            el = np.asarray(el)
            az = np.asarray(az)
            if el.size != az.size:
                raise ValueError("Elevation and azimuth must have the same size")
        return el, az

    def datten(self, el=None, az=None, col_freq="default", troposhpere=True):
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
        if self._interp_d_layers is None:
            raise OrderError(
                "You must calculate the model first. Try running IonModel.calc()"
            )
        el, az = self._check_elaz(el, az)
        h_d = self.d_bot + (self.d_top - self.d_bot) / 2
        delta_h_d = self.d_top - self.d_bot
        d_attenuation = np.empty((el.size, az.size, self.ndlayers))

        if col_freq == "default" or "aggrawal":
            col_model = col_aggarwal
        elif col_freq == "nicolet":
            col_model = col_nicolet
        elif col_freq == "setty":
            col_model = col_setty
        elif col_freq == "average":
            col_model = lambda h: (col_nicolet(h) + col_setty(h)) * 0.5
        else:
            col_model = lambda h: np.float64(col_freq)

        heights = np.linspace(self.d_bot, self.d_top, self.ndlayers)

        theta = np.deg2rad(90 - el)
        if troposhpere:
            theta += trop_refr(theta)

        for i in range(self.ndlayers):
            nu_c = col_model(heights[i])
            ne = self._interp_d_layers[i](el, az)
            ne = np.where(ne > 0, ne, 0)
            plasma_freq = nu_p(ne)
            d_attenuation[:, :, i] = d_atten(
                self.freq, theta, h_d, delta_h_d, plasma_freq, nu_c
            )

        atten = d_attenuation.mean(axis=2)
        if atten.size == 1:
            return atten[0, 0]
        return atten

    def datten_rough(self, el=None, az=None, troposphere=True):
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
        atten = d_atten(self.freq, theta, h_d, delta_h_d, plasma_freq, nu_c)
        if atten.size == 1:
            return atten[0, 0]
        return atten

    def frefr(self, el=None, az=None):
        """
        Calculates refraction in F layer for a given model of ionosphere. Output is the change of zenith angle theta
        (theta -> theta + dtheta). If coordinates are floats the output will be a single number; if they are arrays -
        the output will be a 2D array with dimensions el.size x az.size.

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.

        Returns
        -------
        np.ndarray
        """
        if self._interp_delta_theta is None:
            raise OrderError(
                "You must calculate the model first. Try running IonModel.calc()"
            )
        el, az = self._check_elaz(el, az)
        refr = self._interp_delta_theta(el, az)
        if refr.size == 1:
            return refr[0]
        return refr

    def frefr_rough(self, el=None, az=None):
        """
        Function for rough estimation of the refraction in the F layer if IRI data is not available. See appendix A of
        (Monsalve et al., 2021 ApJ 908 145) for details.

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.

        Returns
        -------
        np.ndarray
        """
        el, az = self._check_elaz(el, az)
        hf = 300
        dhf = 200
        nu_p = 4.49e6
        R_E = 6378100.0
        theta = np.deg2rad(90 - el)
        dtheta = (dhf * (R_E + hf) * nu_p**2 / 3 / R_E**2) * (
            np.sin(theta) * (np.cos(theta) ** 2 + 2 * hf / R_E) ** -1.5 / self.freq**2
        )
        dtheta = np.tile(dtheta, (az.size, 1))
        if dtheta.size == 1:
            return dtheta[0][0]
        return dtheta

    def troprefr(self, el=None, az=None):
        el, az = self._check_elaz(el, az)
        theta = np.deg2rad(90 - el)
        refr = trop_refr(theta)
        refr = np.tile(refr, (az.size, 1))
        if refr.size == 1:
            return refr[0][0]
        return refr

    def save(self, name=None, dir=None):
        """
        # TODO
        """
        import h5py

        if dir == None:
            dir = "calc_results/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is None:
            name = os.path.join(
                dir,
                f"{self.dt.year}_{self.dt.month}_{self.dt.day}_{self.dt.hour}_{self.dt.minute}",
            )
        else:
            name = os.path.join(dir, name)

        if not name.endswith(".h5"):
            name += ".h5"

        with h5py.File(name, mode="w") as file:
            meta = file.create_dataset("meta", shape=(0,))
            meta.attrs["lat0"] = self.lat0
            meta.attrs["lon0"] = self.lon0
            meta.attrs["alt0"] = self.alt0
            meta.attrs["freq"] = self.freq
            meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
            try:
                meta.attrs["ndlayers"] = self.ndlayers
                meta.attrs["d_top"] = self.d_top
                meta.attrs["d_bot"] = self.d_bot
            except TypeError:
                pass
            try:
                meta.attrs["nflayers"] = self.nflayers
                meta.attrs["f_top"] = self.f_top
                meta.attrs["f_bot"] = self.f_bot
            except TypeError:
                pass

            try:
                meta.attrs["gridsize"] = self._gridsize
            except TypeError:
                pass

            try:
                meta.attrs["npoints"] = self._npoints
                file.create_dataset("el", data=self.el)
                file.create_dataset("az", data=self.az)
            except TypeError:
                pass
            try:
                file.create_dataset("d_e_density", data=self._d_e_density)
                file.create_dataset("d_e_temp", data=self._d_e_temp)
                file.create_dataset("d_avg_temp", data=self._d_avg_temp)
            except TypeError:
                pass
            try:
                file.create_dataset("f_e_density", data=self._f_e_density)
                file.create_dataset("phis", data=self._phis)
                file.create_dataset("delta_theta", data=self._delta_theta)
                file.create_dataset("ns", data=self._ns)
            except TypeError:
                pass

    @classmethod
    def load(cls, filename: str):
        import h5py

        if not filename.endswith(".h5"):
            filename += ".h5"
        with h5py.File(filename, mode="r") as file:
            meta = file.get("meta")
            obj = cls(
                lat0=meta.attrs["lat0"],
                lon0=meta.attrs["lon0"],
                alt0=meta.attrs["alt0"],
                freq=meta.attrs["freq"],
                dt=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
            )

            try:
                gridsize = meta.attrs['gridsize']
            except KeyError:
                gridsize = None

            try:
                obj._set_coords(
                    el=_none_or_array(file.get("el")),
                    az=_none_or_array(file.get("az")),
                    gridsize=gridsize,
                )
            except KeyError:
                pass

            try:
                obj.set_lprops(
                    ndlayers=meta.attrs["ndlayers"],
                    d_bot=meta.attrs["d_bot"],
                    d_top=meta.attrs["d_top"],
                    nflayers=meta.attrs["nflayers"],
                    f_bot=meta.attrs["f_bot"],
                    f_top=meta.attrs["f_top"],
                )
            except KeyError:
                pass

            obj._d_e_density = _none_or_array(file.get("d_e_density"))
            obj._d_e_temp = _none_or_array(file.get("d_e_temp"))
            obj._d_avg_temp = _none_or_array(file.get("d_avg_temp"))

            if obj._d_e_density is not None:
                obj._interpolate_d_layer()

            obj._f_e_density = _none_or_array(file.get("f_e_density"))
            obj._phis = _none_or_array(file.get("phis"))
            obj._delta_theta = _none_or_array(file.get("delta_theta"))
            obj._ns = _none_or_array(file.get("ns"))

            if obj._f_e_density is not None:
                obj._interpolate_f_layer()

        return obj

    def plot_dedensity(
        self,
        layer=None,
        interpolated=True,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        if title is None:
            title = r"Average $e^-$ density in D layer"
        if label is None:
            label = r"$m^{-3}$"

        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            if layer is None:
                dd = self._interp_d_aver(el, az)
            elif int(layer) < self.ndlayers:
                dd = self._interp_d_layers[int(layer)](el, az)
            else:
                raise ValueError(
                    "Parameter 'layer' must either be None or int < ndlayers."
                )
            az_rad = az * np.pi / 180.0
            zenith = 90.0 - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot(
                [a, z, dd],
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )
        else:
            if layer is None:
                data = self._d_e_density.mean(axis=1)
            elif int(layer) < self.ndlayers:
                data = self._d_e_density[:, layer]
            else:
                raise ValueError(
                    "Parameter 'layer' must either be None or int < ndlayers."
                )
            return self._polar_plot(
                data,
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )

    def plot_fedensity(
        self,
        interpolated=True,
        layer=None,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        if title is None:
            title = r"Average $e^-$ density in F layer"
        if label is None:
            label = r"$m^{-3}$"
        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            if layer is None:
                dd = self._interp_f_aver(el, az)
            elif int(layer) < self.nflayers:
                dd = self._interp_f_layers[int(layer)](el, az)
            else:
                raise ValueError(
                    "Parameter 'layer' must either be None or int < nflayers."
                )
            az_rad = az * np.pi / 180.0
            zenith = 90.0 - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot(
                [a, z, dd],
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )
        else:
            if layer is None:
                data = self._f_e_density.mean(axis=1)
            elif int(layer) < self.ndlayers:
                data = self._f_e_density[:, layer]
            else:
                raise ValueError(
                    "Parameter 'layer' must either be None or int < nflayers."
                )

            return self._polar_plot(
                data,
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )

    def plot_delta_theta(
        self,
        interpolated=True,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        if title is None:
            title = r"$\delta \theta$ dependence on elevation and azimuth"
        if label is None:
            label = r"$\delta \theta$"

        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            dd = self._interp_delta_theta(el, az)
            az_rad = az * np.pi / 180.0
            zenith = 90.0 - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot(
                [a, z, dd],
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )
        else:
            return self._polar_plot(
                self._delta_theta,
                title=title,
                label=label,
                cblim=cblim,
                file=file,
                dir=dir,
                dpi=dpi,
                cmap=cmap,
            )

    def plot_d_attenuation(
        self,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
        col_freq="default",
    ):
        if title is None:
            title = r"Attenuation factor dependence on elevation and azimuth"
        if label is None:
            label = r"$f$"

        d_attenuation = self.datten(col_freq=col_freq).flatten()
        return self._polar_plot(
            d_attenuation,
            title=title,
            label=label,
            cblim=cblim,
            file=file,
            dir=dir,
            dpi=dpi,
            cmap=cmap,
        )

    def _polar_plot(
        self,
        data,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(111, projection="polar")

        gridsize = int(np.sqrt(self._npoints))
        if gridsize**2 != self._npoints:
            warnings.warn(
                "Can't split data into equal number of rows. "
                "Please make sure your coordinates represent a square grid.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not isinstance(data, list):
            if cblim is None:
                cblim = (np.min(data), np.max(data))
            az_rad = self.az * np.pi / 180.0
            zenith = 90.0 - self.el

            zen_rows = np.split(zenith, gridsize)
            az_rows = np.split(az_rad, gridsize)
            data_rows = np.split(data, gridsize)

            img = ax.pcolormesh(
                az_rows,
                zen_rows,
                data_rows,
                cmap=cmap,
                vmin=cblim[0],
                vmax=cblim[1],
                shading="auto",
            )
        else:
            if cblim is None:
                cblim = (np.min(data[2]), np.max(data[2]))
            img = ax.pcolormesh(
                data[0],
                data[1],
                data[2],
                cmap=cmap,
                vmin=cblim[0],
                vmax=cblim[1],
                shading="auto",
            )
        ax.set_rticks([90, 60, 30, 0])
        ax.scatter(0, 0, c="red", s=5)
        ax.set_theta_zero_location("S")
        plt.colorbar(img).set_label(r"" + label)
        plt.title(title)
        plt.xlabel(datetime.strftime(self.dt, "%Y-%m-%d %H:%M"))

        if file is not None:
            if dir == None:
                dir = "pictures/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(os.path.join(dir, file), dpi=dpi)
            plt.close(fig)
            return
        return fig
