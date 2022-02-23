import os
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import itertools as it
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import iri2016 as ion
import pymap3d as pm
from tqdm import tqdm

from datetime import datetime, timedelta
from time import time


class OrderError(Exception):
    """
    Exception indicating incorrect order of simulation routines.
    """
    pass


def check_latlon(lat, lon):
    if not -90 <= lat <= 90:
        raise ValueError("Latitude of the instrument must be in range [-90, 90]")
    if not -180 <= lon < 180:
        raise ValueError("Longitude of the instrument must be in range [-180, 180]")


def srange(theta, alt, RE=6378000.):
    """
    Calculates the distance in meters from the telescope to the point (theta, alt).

    Parameters
    ----------
    theta : float
        Zenith angle in radians
    alt : float
        Altitude in meters
    RE : float, optional
        Radius of the Earth in meters

    Returns
    -------
    r : float
        Range in meters
    """
    r = -RE * np.cos(theta) + np.sqrt((RE * np.cos(theta)) ** 2 + alt ** 2 + 2 * alt * RE)
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


def nu_p(n_e):
    """
    Plasma frequency from electron density

    Parameters
    ----------
    n_e : float
        Electron density

    Returns
    -------
    float
        Plasma frequency in Hz
    """
    e = 1.60217662e-19
    m_e = 9.10938356e-31
    epsilon0 = 8.85418782e-12
    if n_e < 0:
        raise ValueError('Number density cannot be < 0.')
    return 1 / (2 * np.pi) * np.sqrt((n_e * e ** 2) / (m_e * epsilon0))


def n_f(n_e, freq):
    """
    Refractive index of F-layer from electron density

    Parameters
    ----------
    n_e : float
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
    n1 : float
        Refractive index in previous medium
    n2 : float
        Refractive index in current medium
    phi : float
        Angle of incident ray in rad

    Returns
    -------
    float
        Angle in rad
    """
    return np.arcsin(n1 / n2 * np.sin(phi))


def d_atten(nu, theta, h_d, delta_hd, nu_p, nu_c):
    """
    #TODO
    """
    R_E = 6371000
    c = 2.99792458e8
    delta_s = delta_hd * (1 + h_d / R_E) * (np.cos(theta) ** 2 + 2 * h_d / R_E) ** (-0.5)
    f = np.exp(-(2 * np.pi * nu_p ** 2 * nu_c * delta_s) / (c * (nu_c ** 2 + nu ** 2)))
    return f


def _d_temp_density(dt, d_bot, d_top, ndlayers, el, az, lat, lon, alt):
    """
    # TODO
    """
    d_heights = np.linspace(d_bot, d_top, ndlayers)
    d_srange = np.empty(ndlayers)

    for i in range(ndlayers):
        d_srange[i] = srange((90 - el) * np.pi / 180, d_heights[i])

    d_obs_lat = np.empty(ndlayers)
    d_obs_lon = np.empty(ndlayers)
    d_obs_h = np.empty(ndlayers)

    for i in range(ndlayers):
        d_obs_lat[i], d_obs_lon[i], d_obs_h[i] = pm.aer2geodetic(
            az, el, d_srange[i], lat, lon, alt)

    d_e_density = np.empty(ndlayers)
    d_e_temp = np.empty(ndlayers)

    for i in range(ndlayers):
        alt_range = [d_obs_h[i] / 1000, d_obs_h[i] / 1000, 1]
        alt_prof = ion.IRI(dt, alt_range, d_obs_lat[i], d_obs_lon[i])

        if alt_prof.ne.data > 0:
            d_e_density[i] = alt_prof.ne.data
        else:
            d_e_density[i] = 0

        d_e_temp[i] = alt_prof.Te.data

    # return d_e_density, d_e_temp
    return d_e_density, d_e_temp, d_obs_lat, d_obs_lon


def _d_temp_density_star(pars):
    return _d_temp_density(*pars)


def _calc_flayer(dt, f_bot, f_top, nflayers, el, az, lat, lon, alt, freq):
    """
    #TODO
    """
    R_E = 6371000.
    f_heights = np.linspace(f_bot, f_top, nflayers)
    ns = np.empty(nflayers)  # refractive indices
    f_e_density = np.empty(nflayers)
    phis = np.empty(nflayers)  # angles of refraction
    delta_phi = 0.  # total change in angle

    # Distance from telescope to first layer
    r_slant = srange((90. - el) * np.pi / 180., f_heights[0] - alt)
    # Geodetic coordinates of 'hit point' on the first layer
    lat_ray, lon_ray, h_ray = pm.aer2geodetic(az, el, r_slant, lat, lon, alt)

    # The sides of the 1st triangle
    d_tel = R_E + alt  # Distance from Earth center to telescope
    d_cur = R_E + h_ray  # Distance from Earth center to layer

    # The inclination angle at the 1st interface using law of cosines [rad]
    cosphi_inc = (r_slant ** 2 + d_cur ** 2 - d_tel ** 2) / (2 * r_slant * d_cur)
    assert cosphi_inc <= 1, "Something is wrong with coordinates."
    phi_inc = np.arccos(cosphi_inc)

    # Refraction index of air
    n_cur = 1.

    # Get IRI info of point
    # f_alt_prof = ion.IRI(dt, [h_cur / 1e3, h_cur / 1e3, 1], lat, lon)
    f_alt_prof = ion.IRI(dt, [h_ray / 1e3, h_ray / 1e3, 1], lat, lon)
    f_e_density[0] = f_alt_prof.ne.data[0]

    # Refraction index of 1st point
    n_next = n_f(f_e_density[0], freq)
    ns[0] = n_next

    # The outgoing angle at the 1st interface using Snell's law
    phi_ref = refr_angle(n_cur, n_next, phi_inc)
    phis[0] = phi_ref

    # #TODO: check if correct
    delta_phi += (phi_ref - phi_inc)

    # el_cur = el - (phi_ref - phi_inc)
    el_cur = np.rad2deg(np.pi / 2 - phi_ref)

    n_cur = n_next

    for i in range(1, nflayers):
        h_next = f_heights[i]
        d_next = R_E + h_next

        # Angle between d_cur and r_slant
        int_angle = np.pi - phi_ref
        # The inclination angle at the i-th interface using law of sines [rad]
        phi_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

        # Getting r2 using law of cosines
        # r_slant = d_cur * np.cos(int_angle) + np.sqrt(d_next ** 2 - d_cur ** 2 * np.sin(int_angle) ** 2)
        # r_slant = srange((90. - el_cur) * np.pi / 180., d_next - d_cur)
        r_slant = srange((90. - el_cur) * np.pi / 180., d_next - d_cur, RE=R_E + d_cur)
        # Get geodetic coordinates of point
        lat_ray, lon_ray, h_ray = pm.aer2geodetic(az, el_cur, r_slant, lat_ray, lon_ray, h_ray)

        # Get IRI info of 2nd point
        f_alt_prof = ion.IRI(dt, [h_ray / 1000, h_ray / 1000, 1], lat_ray, lon_ray)
        f_e_density[i] = f_alt_prof.ne.data[0]
        if f_e_density[i] < 0:
            raise ValueError("Something went wrong. Number density cannot be < 0.")

        # Refractive indices
        n_next = n_f(f_e_density[i], freq)
        ns[i] = n_next

        # If this is the last point then use refractive index of vacuum
        if i == nflayers - 1:
            n_next = 1

        # The outgoing angle at the 2nd interface using Snell's law
        phi_ref = refr_angle(n_cur, n_next, phi_inc)
        phis[i] = phi_ref
        delta_phi += (phi_ref - phi_inc)

        # Update variables for new interface
        # el_cur = el_cur - (phi_ref - phi_inc)
        el_cur = np.rad2deg(np.pi / 2 - phi_ref)
        n_cur = n_next
        d_cur = d_next

    return f_e_density, phis, delta_phi, ns


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

        self.npoints = None
        self.gridsize = None
        self.el = None
        self.az = None

        self.ndlayers = None
        self.d_bot = None
        self.d_top = None

        self.d_obs_h = None
        self.d_e_density = None
        self.d_e_temp = None
        self.d_attenuation = None
        self.d_avg_temp = None
        self.d_col_freq = None

        self.nflayers = None
        self.f_bot = None
        self.f_top = None

        self.f_e_density = None
        self.phis = None
        self.delta_phi = None
        self.ns = None

        self.interp_d_layers = None
        self.interp_d_aver = None
        self.interp_f_layers = None
        self.interp_f_aver = None
        self.interp_delta_phi = None

    def set_coords(self, el, az, gridsize=None):
        if len(el) != len(az):
            raise ValueError("Elevation and azimuth must be the same length.")
        self.el = el
        self.az = az
        self.gridsize = gridsize
        self.npoints = len(el)

    def generate_coord_grid(self, el_start=0., el_end=90., az_start=0., az_end=360., gridsize=32):
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

        self.az = np.repeat(az_vals, gridsize)
        self.el = np.tile(el_vals, gridsize)
        self.gridsize = gridsize
        self.npoints = gridsize * gridsize

    def setup_dlayer(self, nlayers=10, d_bot=6e4, d_top=9e4, col_freq='default'):
        """
        Set up all necessary parameters for the D-layer of the ionosphere

        Parameters
        ----------
        nlayers : int
            Number of layers in D-layer
        d_bot : float
            Lower limit of the D-layer in meters
        d_top : float
            Upper limit of the D-layer in meters
        col_freq : str, float
            The collision frequency ('default', 'nicolet', 'setty', 'average' or float in Hz)
        """
        self.ndlayers = nlayers
        self.d_bot = d_bot
        self.d_top = d_top
        self.d_col_freq = col_freq

    def setup_flayer(self, nlayers=40, f_bot=1.5e5, f_top=5e5):
        """
        Set up all necessary parameters for the F-layer of the ionosphere

        Parameters
        ----------
        nlayers : int
            Number of layers in F-layer
        f_bot : float
            Lower limit of the F-layer in meters
        f_top : float
            Upper limit of the F-layer in meters
        """
        self.nflayers = nlayers
        self.f_bot = f_bot
        self.f_top = f_top

    def _interpolate_d_layer(self, kind='cubic'):
        from scipy.interpolate import interp2d
        az_vals = np.linspace(np.min(self.az), np.max(self.az), self.gridsize, endpoint=True)
        el_vals = np.linspace(np.min(self.el), np.max(self.el), self.gridsize)
        lmodels = [
            interp2d(el_vals, az_vals, self.d_e_density[:, i].reshape(self.gridsize, self.gridsize), kind=kind)
            for i in range(self.ndlayers)
        ]
        self.interp_d_layers = lmodels
        aver_data = self.d_e_density.mean(axis=1)
        self.interp_d_aver = interp2d(el_vals, az_vals, aver_data.reshape(self.gridsize, self.gridsize), kind=kind)

    def _interpolate_f_layer(self, kind='cubic'):
        from scipy.interpolate import interp2d
        az_vals = np.linspace(np.min(self.az), np.max(self.az), self.gridsize, endpoint=True)
        el_vals = np.linspace(np.min(self.el), np.max(self.el), self.gridsize)
        lmodels = [
            interp2d(el_vals, az_vals, self.f_e_density[:, i].reshape(self.gridsize, self.gridsize), kind=kind)
            for i in range(self.nflayers)
        ]
        self.interp_f_layers = lmodels
        aver_data = self.f_e_density.mean(axis=1)
        self.interp_f_aver = interp2d(el_vals, az_vals, aver_data.reshape(self.gridsize, self.gridsize), kind=kind)

    def _interpolate_delta_phi(self, kind='cubic'):
        from scipy.interpolate import interp2d
        az_vals = np.linspace(np.min(self.az), np.max(self.az), self.gridsize, endpoint=True)
        el_vals = np.linspace(np.min(self.el), np.max(self.el), self.gridsize)
        self.interp_delta_phi = interp2d(el_vals, az_vals, self.delta_phi.reshape(self.gridsize, self.gridsize), kind=kind)

    def calc(self, processes=1, progressbar=False, layer=None):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density(d_e_density),
        electron temperature(d_e_temp), attenuation factor(d_attenuation), average temperature(d_avg_temp)] and
        F-layer [electron density(f_e_density), angle of the outgoing refracted beam at each layer(phis),
        the net deviation of the elevation angle for each coordinate(delta_phi), refractive index at each layer(ns)].
        """

        if None in [self.ndlayers, self.d_bot, self.d_top] and layer != 'f' and layer != 'F':
            raise OrderError("You have to set up parameters for the D layer first (use the setup_dlayer() method)")

        if None in [self.nflayers, self.f_bot, self.f_top] and layer != 'd' and layer != 'D':
            raise OrderError("You have to set up parameters for the F layer first (use the setup_flayer() method)")

        cpus = cpu_count()
        if cpus < processes:
            processes = cpus
            warnings.warn(
                f"You have only {cpus} cpu threads available. Setting number of processes to {cpus}.",
                RuntimeWarning,
                stacklevel=2,
            )

        if layer != 'f' and layer != 'F':
            if not progressbar:
                print("Starting calulation for D layer for date " + str(self.dt), flush=True)
                t1_d = time()
            with Pool(processes=processes) as pool:
                dlayer = list(tqdm(pool.imap(
                    _d_temp_density_star,
                    zip(
                        it.repeat(self.dt),
                        it.repeat(self.d_bot),
                        it.repeat(self.d_top),
                        it.repeat(self.ndlayers),
                        self.el,
                        self.az,
                        it.repeat(self.lat0),
                        it.repeat(self.lon0),
                        it.repeat(self.alt0),
                    )),
                    total=self.npoints,
                    disable=not progressbar,
                    desc='D-layer',
                ))
                self.d_e_density = np.vstack([d[0] for d in dlayer])
                self.d_e_temp = np.vstack([d[1] for d in dlayer])
                self.d_obs_lat = np.vstack([d[2] for d in dlayer])
                self.d_obs_lon = np.vstack([d[3] for d in dlayer])
            self._calc_d_attenuation()
            self._calc_d_avg_temp()
            self._interpolate_d_layer()
            if not progressbar:
                print(f"Calulation for D layer have ended with {time() - t1_d:.1f} seconds.", flush=True)

        if layer != 'd' and layer != 'D':
            if not progressbar:
                print("Starting calulation for F layer for date " + str(self.dt), flush=True)
                t1_f = time()
            with Pool(processes=processes) as pool:
                flayer = list(tqdm(pool.imap(
                    _calc_flayer_star,
                    zip(
                        it.repeat(self.dt),
                        it.repeat(self.f_bot),
                        it.repeat(self.f_top),
                        it.repeat(self.nflayers),
                        self.el,
                        self.az,
                        it.repeat(self.lat0),
                        it.repeat(self.lon0),
                        it.repeat(self.alt0),
                        it.repeat(self.freq)
                    )),
                    total=self.npoints,
                    disable=not progressbar,
                    desc='F-layer',
                ))
                self.f_e_density = np.vstack([f[0] for f in flayer])
                self.phis = np.vstack([f[1] for f in flayer])
                self.delta_phi = np.vstack([f[2] for f in flayer]).reshape([-1])
                self.ns = np.vstack([f[3] for f in flayer])
            self._interpolate_f_layer()
            if not progressbar:
                print(f"Calulation for F layer have ended with {time() - t1_f:.1f} seconds.", flush=True)

        return

    def _calc_d_attenuation(self):
        """
        Calculates the attenuation factor from frequency of the signal [Hz], angle [rad],
        altitude of the D-layer midpoint [km], thickness of the D-layer [km], plasma frequency [Hz],
        and electron collision frequency [Hz]. Output is the attenuation factor between 0 (total attenuation)
        and 1 (no attenuation).
        """
        h_d = self.d_bot + (self.d_top - self.d_bot) / 2
        delta_h_d = self.d_top - self.d_bot
        d_attenuation = np.empty(self.npoints)

        if self.d_col_freq in ['nicolet', 'setty', 'average']:
            if self.d_col_freq == 'nicolet':
                col_model = col_nicolet
            elif self.d_col_freq == 'setty':
                col_model = col_setty
            else:
                col_model = lambda h: (col_nicolet(h) + col_setty(h)) * 0.5

            for i in range(self.npoints):
                nu_c = col_model(self.d_obs_h[i] / 1000)
                d_attenuation_temp = np.empty(self.ndlayers)

                for j in range(self.ndlayers):
                    plasma_freq = nu_p(self.d_e_density[i][j])
                    d_attenuation_temp[j] = d_atten(self.freq, (90 - self.el[i]) * np.pi / 180, h_d, delta_h_d,
                                                    plasma_freq,
                                                    nu_c[j])

                d_attenuation[i] = np.average(d_attenuation_temp)

        else:
            if self.d_col_freq == 'default':
                nu_c = 10e6
            else:
                nu_c = np.float64(self.d_col_freq)

            d_avg_density = np.empty(self.npoints)

            for i in range(self.npoints):
                d_avg_density[i] = np.average(self.d_e_density[i])
                plasma_freq = nu_p(d_avg_density[i])
                d_attenuation[i] = d_atten(self.freq, (90 - self.el[i]) * np.pi / 180, h_d, delta_h_d, plasma_freq,
                                           nu_c)

        self.d_attenuation = d_attenuation

    def _calc_d_avg_temp(self):
        """
        Calculates average temperature for the D-layer
        """
        d_avg_temp = np.empty(self.npoints)
        for i in range(self.npoints):
            temp = self.d_e_temp[i][self.d_e_temp[i] > 0]
            if len(temp) > 0:
                d_avg_temp[i] = np.average(temp)
            else:
                d_avg_temp[i] = 0

        self.d_avg_temp = d_avg_temp

    def save(self, name=None, dir=None):
        """
        # TODO
        """
        import h5py
        if dir == None:
            dir = 'calc_results/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is None:
            name = os.path.join(dir, f"{self.dt.year}_{self.dt.month}_{self.dt.day}_{self.dt.hour}_{self.dt.minute}")
        else:
            name = os.path.join(dir, name)

        if not name.endswith('.h5'):
            name += '.h5'

        with h5py.File(name, mode='w') as file:
            meta = file.create_dataset('meta', shape=(0,))
            meta.attrs['lat0'] = self.lat0
            meta.attrs['lon0'] = self.lon0
            meta.attrs['alt0'] = self.alt0
            meta.attrs['freq'] = self.freq
            meta.attrs['dt'] = self.dt.strftime('%Y-%m-%d %H:%M')
            try:
                meta.attrs['ndlayers'] = self.ndlayers
                meta.attrs['d_top'] = self.d_top
                meta.attrs['d_bot'] = self.d_bot
            except TypeError:
                pass
            try:
                meta.attrs['nflayers'] = self.nflayers
                meta.attrs['f_top'] = self.f_top
                meta.attrs['f_bot'] = self.f_bot
            except TypeError:
                pass

            try:
                meta.attrs['gridsize'] = self.gridsize
            except TypeError:
                pass

            try:
                meta.attrs['npoints'] = self.npoints
                file.create_dataset('el', data=self.el)
                file.create_dataset('az', data=self.az)
            except TypeError:
                pass
            try:
                file.create_dataset('d_e_density', data=self.d_e_density)
                file.create_dataset('d_e_temp', data=self.d_e_temp)
                file.create_dataset('d_attenuation', data=self.d_attenuation)
                file.create_dataset('d_avg_temp', data=self.d_avg_temp)
            except TypeError:
                pass
            try:
                file.create_dataset('f_e_density', data=self.f_e_density)
                file.create_dataset('phis', data=self.phis)
                file.create_dataset('delta_phi', data=self.delta_phi)
                file.create_dataset('ns', data=self.ns)
            except TypeError:
                pass

    @classmethod
    def load(cls, filename: str):
        import h5py
        if not filename.endswith('.h5'):
            filename += '.h5'
        with h5py.File(filename, mode='r') as file:
            meta = file.get('meta')
            obj = cls(
                lat0=meta.attrs['lat0'],
                lon0=meta.attrs['lon0'],
                alt0=meta.attrs['alt0'],
                freq=meta.attrs['freq'],
                dt=datetime.strptime(meta.attrs['dt'], '%Y-%m-%d %H:%M'),
            )
            try:
                gridsize = meta.attrs['gridsize']
            except KeyError:
                gridsize = None
            try:
                obj.set_coords(
                    el=np.array(file.get('el')),
                    az=np.array(file.get('az')),
                    gridsize=gridsize,
                )
                obj.setup_dlayer(
                    nlayers=meta.attrs['ndlayers'],
                    d_bot=meta.attrs['d_bot'],
                    d_top=meta.attrs['d_top'],
                )
            except KeyError:
                pass

            try:
                obj.setup_flayer(
                    nlayers=meta.attrs['nflayers'],
                    f_bot=meta.attrs['f_bot'],
                    f_top=meta.attrs['f_top'],
                )
            except KeyError:
                pass

            obj.d_e_density = np.array(file.get('d_e_density'))
            obj.d_e_temp = np.array(file.get('d_e_temp'))
            obj.d_attenuation = np.array(file.get('d_attenuation'))
            obj.d_avg_temp = np.array(file.get('d_avg_temp'))

            if obj.d_e_density is not None:
                obj._interpolate_d_layer()

            obj.f_e_density = np.array(file.get('f_e_density'))
            obj.phis = np.array(file.get('phis'))
            obj.delta_phi = np.array(file.get('delta_phi'))
            obj.ns = np.array(file.get('ns'))

            if obj.f_e_density is not None:
                obj._interpolate_f_layer()
            if obj.delta_phi is not None:
                obj._interpolate_delta_phi()

        return obj

    def plot_dedensity(self, layer=None, interpolated=True, title=None, label=None, cblim=None, file=None, dir=None,
                       dpi=300, cmap='viridis'):
        if title is None:
            title = r'Average $e^-$ density in D layer'
        if label is None:
            label = r"$m^{-3}$"

        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            if layer is None:
                dd = self.interp_d_aver(el, az)
            elif int(layer) < self.ndlayers:
                dd = self.interp_d_layers[int(layer)](el, az)
            else:
                raise ValueError("Parameter 'layer' must either be None or int < ndlayers.")
            az_rad = az * np.pi / 180.
            zenith = 90. - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot([a, z, dd], title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi,
                                    cmap=cmap)
        else:
            if layer is None:
                data = self.d_e_density.mean(axis=1)
            elif int(layer) < self.ndlayers:
                data = self.d_e_density[:, layer]
            else:
                raise ValueError("Parameter 'layer' must either be None or int < ndlayers.")

            return self._polar_plot(data, title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi, cmap=cmap)

    def plot_fedensity(self, interpolated=True, layer=None, title=None, label=None, cblim=None, file=None, dir=None,
                       dpi=300, cmap='viridis'):
        if title is None:
            title = r'Average $e^-$ density in F layer'
        if label is None:
            label = r"$m^{-3}$"
        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            if layer is None:
                dd = self.interp_f_aver(el, az)
            elif int(layer) < self.nflayers:
                dd = self.interp_f_layers[int(layer)](el, az)
            else:
                raise ValueError("Parameter 'layer' must either be None or int < nflayers.")
            az_rad = az * np.pi / 180.
            zenith = 90. - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot([a, z, dd], title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi,
                                    cmap=cmap)
        else:
            if layer is None:
                data = self.f_e_density.mean(axis=1)
            elif int(layer) < self.ndlayers:
                data = self.f_e_density[:, layer]
            else:
                raise ValueError("Parameter 'layer' must either be None or int < nflayers.")

            return self._polar_plot(data, title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi, cmap=cmap)

    def plot_delta_phi(self, interpolated=True, title=None, label=None, cblim=None, file=None, dir=None,
                       dpi=300, cmap='viridis'):
        if title is None:
            title = r'$\delta \theta$ dependence on elevation and azimuth'
        if label is None:
            label = r"$\delta \theta$"

        if interpolated:
            grsz = 1000
            az = np.linspace(np.min(self.az), np.max(self.az), grsz, endpoint=True)
            el = np.linspace(np.min(self.el), np.max(self.el), grsz)
            dd = self.interp_delta_phi(el, az)
            az_rad = az * np.pi / 180.
            zenith = 90. - el
            z, a = np.meshgrid(zenith, az_rad)
            return self._polar_plot([a, z, dd], title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi,
                                    cmap=cmap)
        else:
            return self._polar_plot(self.delta_phi, title=title, label=label, cblim=cblim, file=file, dir=dir, dpi=dpi, cmap=cmap)

    def _polar_plot(self, data, title=None, label=None, cblim=None, file=None, dir=None, dpi=300, cmap='viridis'):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(111, projection='polar')

        gridsize = int(np.sqrt(self.npoints))
        if gridsize ** 2 != self.npoints:
            warnings.warn(
                "Can't split data into equal number of rows. "
                "Please make sure your coordinates represent a square grid.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not isinstance(data, list):
            if cblim is None:
                cblim = (np.min(data), np.max(data))
            az_rad = self.az * np.pi / 180.
            zenith = 90. - self.el

            zen_rows = np.split(zenith, gridsize)
            az_rows = np.split(az_rad, gridsize)
            data_rows = np.split(data, gridsize)

            img = ax.pcolormesh(az_rows, zen_rows, data_rows, cmap=cmap, vmin=cblim[0], vmax=cblim[1], shading='auto')
        else:
            if cblim is None:
                cblim = (np.min(data[2]), np.max(data[2]))
            img = ax.pcolormesh(data[0], data[1], data[2], cmap=cmap, vmin=cblim[0], vmax=cblim[1], shading='auto')
        ax.set_rticks([90, 60, 30, 0])
        ax.scatter(0, 0, c='red', s=5)
        ax.set_theta_zero_location("S")
        plt.colorbar(img).set_label(r'' + label)
        plt.title(title)
        plt.xlabel(datetime.strftime(self.dt, '%Y-%m-%d %H:%M'))

        if file is not None:
            if dir == None:
                dir = 'pictures/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(os.path.join(dir, file), dpi=dpi)
            plt.close(fig)
            return
        return fig
