import warnings
from multiprocessing import Pool, cpu_count
import itertools

import iricore
import numpy as np
import pymap3d as pm
from tqdm import tqdm

from mistion.modules.helpers import check_latlon, Ellipsoid, generate_grid, OrderError
from mistion.modules.ion_tools import srange, n_f, refr_angle


def _calc_flayer(dt, f_bot, f_top, nflayers, els, azs, lat, lon, alt, freq):
    """
    #TODO
    """
    R_E = 6378100.0
    ell = Ellipsoid()
    ncoords = len(els)
    f_heights = np.linspace(f_bot, f_top, nflayers) * 1000
    f_e_density = np.empty((ncoords, nflayers))
    f_e_temp = np.empty((ncoords, nflayers))
    delta_theta = np.zeros(ncoords)  # total change in angle (ncoords, nflayers)

    # Distance from telescope to first layer
    r_slant = srange((90 - els) * np.pi / 180, f_heights[0] - alt)
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
        dt, [f_bot * 1e-3, f_bot * 1e-3, 1], lat_ray, lon_ray, replace_missing=0
    )
    f_e_density[:, 0] = f_alt_prof["ne"][:, 0]
    f_e_temp[:, 0] = f_alt_prof["te"][:, 0]

    # Refraction index of 1st point
    n_next = n_f(f_e_density[:, 0], freq)
    # The outgoing angle at the 1st interface using Snell's law
    theta_ref = refr_angle(n_cur, n_next, theta_inc)
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
        f_e_temp[:, i] = f_alt_prof["te"][:, 0]

        # Refractive indices
        n_next = n_f(f_e_density[:, i], freq)

        # # If this is the last point then use refractive index of vacuum
        # if i == nflayers - 1:
        #     n_next = np.ones(ncoords)

        # The outgoing angle at the 2nd interface using Snell's law
        theta_ref = refr_angle(n_cur, n_next, theta_inc)
        delta_theta += theta_ref - theta_inc

        # Update variables for new interface
        el_cur = np.rad2deg(np.pi / 2 - theta_ref)
        n_cur = n_next
        d_cur = d_next

    return f_e_density, f_e_temp, delta_theta


def _calc_flayer_star(pars):
    return _calc_flayer(*pars)


class FLayer:
    def __init__(
        self,
        dt,
        position,
        freq,
        fbot=60,
        ftop=90,
        nflayers=30,
        elrange=None,
        azrange=None,
        gridsize=50,
    ):
        check_latlon(position[0], position[1])
        self.fbot = fbot
        self.ftop = ftop
        self.nflayers = nflayers
        self.dt = dt
        self.freq = freq
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

        self._f_e_density = None
        self._f_e_temp = None
        self._dtheta = None

        self._interp_fed = None
        self._interp_feda = None
        self._interp_fet = None
        self._interp_feta = None
        self._interp_dtheta = None

    def _interpolate_f_layer(self, kind="cubic"):
        from scipy.interpolate import interp2d

        az_vals = np.linspace(*self.azrange, self.gridsize, endpoint=True)
        el_vals = np.linspace(*self.elrange, self.gridsize)
        self._interp_fed = [
            interp2d(
                el_vals,
                az_vals,
                self._f_e_density[:, i].reshape(self.gridsize, self.gridsize),
                kind=kind,
            )
            for i in range(self.nflayers)
        ]
        self._interp_fet = [
            interp2d(
                el_vals,
                az_vals,
                self._f_e_temp[:, i].reshape(self.gridsize, self.gridsize),
                kind=kind,
            )
            for i in range(self.nflayers)
        ]
        aver_data = self._f_e_density.mean(axis=1)
        self._interp_feda = interp2d(
            el_vals,
            az_vals,
            aver_data.reshape(self.gridsize, self.gridsize),
            kind=kind,
        )
        aver_data = self._f_e_temp.mean(axis=1)
        self._interp_feta = interp2d(
            el_vals,
            az_vals,
            aver_data.reshape(self.gridsize, self.gridsize),
            kind=kind,
        )
        self._interp_dtheta = interp2d(
            el_vals,
            az_vals,
            self._dtheta.reshape(self.gridsize, self.gridsize),
            kind=kind,
        )

    def calc(self, nproc=1, pbar=True, batch=500):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density, electron
        temperature, attenuation factor, average temperature] and F-layer [electron density, angle of the outgoing
        refracted beam at each layer, the net deviation of the elevation angle for each coordinate, refractive index
        at each layer].
        """

        el, az = generate_grid(
            *self.elrange, *self.azrange, self.gridsize
        )
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
            flayer = list(
                tqdm(
                    pool.imap(
                        _calc_flayer_star,
                        zip(
                            itertools.repeat(self.dt),
                            itertools.repeat(self.fbot),
                            itertools.repeat(self.ftop),
                            itertools.repeat(self.nflayers),
                            el_batches,
                            az_batches,
                            itertools.repeat(self.lat0),
                            itertools.repeat(self.lon0),
                            itertools.repeat(self.alt0),
                            itertools.repeat(self.freq),
                        ),
                    ),
                    total=len(el_batches),
                    disable=not pbar,
                    desc="F-layer",
                )
            )
            self._f_e_density = np.vstack([f[0] for f in flayer])
            self._f_e_temp = np.vstack([f[1] for f in flayer])
            self._dtheta = np.hstack([f[2] for f in flayer]).reshape([-1])
        self._interpolate_f_layer()
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

    def frefr(self, el=None, az=None):
        """
        Calculates refraction in F layer for a given model of ionosphere. Output is the change of zenith angle theta
        (theta -> theta + dtheta). If coordinates are floats the output will be a single number; if they are arrays -
        the output will be a 2D array with dimensions az.size x el.size (according to np.meshgrid(el, az)).

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.

        Returns
        -------
        dtheta : float : np.ndarray
            Change in elevation in degrees
        """
        if self._interp_dtheta is None:
            raise OrderError(
                "You must calculate the model first. Try running IonModel.calc()"
            )
        el, az = self._check_elaz(el, az, size_err=False)
        refr = self._interp_dtheta(el, az)
        if refr.size == 1:
            return np.rad2deg(refr[0])
        return np.rad2deg(refr)

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
        dtheta : float : np.ndarray
            Change in elevation in degrees
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
            return np.rad2deg(dtheta[0][0])
        return np.rad2deg(dtheta)
