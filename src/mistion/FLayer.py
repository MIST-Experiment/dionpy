import itertools
from multiprocessing import Pool, cpu_count

import healpy as hp
import iricore
import numpy as np
import pymap3d as pm
from tqdm import tqdm

from mistion.modules.helpers import Ellipsoid, eval_layer, iri_star, check_elaz_shape
from mistion.modules.ion_tools import srange, n_f, refr_angle, trop_refr


class FLayer:
    """
    A model of the F layer of the ionosphere. Includes electron density and temperature data after calculation
    and implements a model of ionospheric refraction.

    :param dt: Date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param fbot: Lower limit in [km] of the F layer of the ionosphere.
    :param ftop: Upper limit in [km] of the F layer of the ionosphere.
    :param nflayers: Number of sub-layers in the F layer for intermediate calculations.
    :param nside: Resolution of healpix grid.
    :param pbar: If True - a progress bar will appear.
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """
    def __init__(
            self,
            dt,
            position,
            fbot=150,
            ftop=500,
            nflayers=30,
            nside=128,
            pbar: bool = True,
            _autocalc: bool = True,
    ):
        self.fbot = fbot
        self.ftop = ftop
        self.nflayers = nflayers
        self.dt = dt
        self.position = position

        self.nside = nside
        self._rdeg = 24  # radius of disc queried to healpy
        self._posvec = hp.ang2vec(self.position[1], self.position[0], lonlat=True)
        self._obs_pixels = hp.query_disc(
            self.nside, self._posvec, np.deg2rad(self._rdeg), inclusive=True
        )
        self._obs_lons, self._obs_lats = hp.pix2ang(
            self.nside, self._obs_pixels, lonlat=True
        )

        self.f_e_density = np.zeros((len(self._obs_pixels), nflayers))
        self.f_e_temp = np.zeros((len(self._obs_pixels), nflayers))
        if _autocalc:
            self._calc_par(pbar=pbar)

    def _calc(self):
        """
        Makes a single call to iricore (assuming already implemented parallelism) requesting
        electron density and electron temperature for future use in attenuation modeling.
        """
        f_heights = np.linspace(self.fbot, self.ftop, self.nflayers)
        for i in tqdm(range(self.nflayers)):
            res = iricore.IRI(
                self.dt,
                [f_heights[i], f_heights[i], 1],
                self._obs_lats,
                self._obs_lons,
                replace_missing=0,
            )
            self.f_e_density[:, i] = res["ne"][:, 0]
            self.f_e_temp[:, i] = res["te"][:, 0]
        return

    def _calc_par(self, pbar=True):
        """
        Makes several calls to iricore in parallel requesting electron density and
        electron temperature for future use in attenuation modeling.
        """
        batch = 1000
        nbatches = len(self._obs_pixels) // batch + 1
        nproc = np.min([cpu_count(), nbatches])
        blat = np.array_split(self._obs_lats, nbatches)
        blon = np.array_split(self._obs_lons, nbatches)
        heights = (self.fbot, self.ftop, (self.ftop - self.fbot) / (self.nflayers - 1) - 1e-6)

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
                    desc="F layer",
                )
            )
            self.f_e_density = np.vstack([r["ne"] for r in res])
            self.f_e_temp = np.vstack([r["te"] for r in res])
        return

    def fed(self, el, az, layer=None):
        """
        :param el: Elevation of an observation.
        :param az: Azimuth of an observation.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron density in the F layer.
        """
        return eval_layer(
            el,
            az,
            self.nside,
            self.position,
            self.fbot,
            self.ftop,
            self.nflayers,
            self._obs_pixels,
            self.f_e_density,
            layer=layer,
        )

    def fet(self, el, az, layer=None):
        """
        :param el: Elevation of an observation.
        :param az: Azimuth of an observation.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron temperature in the F layer.
        """
        return eval_layer(
            el,
            az,
            self.nside,
            self.position,
            self.fbot,
            self.ftop,
            self.nflayers,
            self._obs_pixels,
            self.f_e_temp,
            layer=layer,
        )

    def refr(self, el, az, freq, troposphere=True):
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param freq: Frequency of observation(s) in [MHz]. If  - the calculation will be performed in parallel on all
                     available cores. Requires `dt` to be a single datetime object.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :return: Refraction angle in [deg] at given sky coordinates, time and frequency of observation.
        """
        check_elaz_shape(el, az)
        el, az = el.copy(), az.copy()
        re = 6378100.0
        ell = Ellipsoid()
        f_heights = np.linspace(self.fbot, self.ftop, self.nflayers) * 1e3
        delta_theta = 0 * el

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
        costheta_inc = (r_slant ** 2 + d_cur ** 2 - d_tel ** 2) / (2 * r_slant * d_cur)
        assert (costheta_inc <= 1).all(), "Something is wrong with coordinates."
        theta_inc = np.arccos(costheta_inc)

        # Refraction index of air
        n_cur = np.ones(el.shape)

        # Get IRI info of point
        fed = self.fed(el, az, layer=0)

        # Refraction index of 1st point
        n_next = n_f(fed, freq)
        # The outgoing angle at the 1st interface using Snell's law
        theta_ref = refr_angle(n_cur, n_next, theta_inc)
        delta_theta += theta_ref - theta_inc
        el_cur = np.rad2deg(np.pi / 2 - theta_ref)
        n_cur = n_next

        for i in range(1, self.nflayers):
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
            # Get IRI info of 2nd point
            fed = self.fed(el, az, layer=i)

            # Refractive indices
            n_next = n_f(fed, freq)

            # The outgoing angle at the 2nd interface using Snell's law
            theta_ref = refr_angle(n_cur, n_next, theta_inc)
            delta_theta += theta_ref - theta_inc

            # Update variables for new interface
            el_cur = np.rad2deg(np.pi / 2 - theta_ref)
            n_cur = n_next
            d_cur = d_next

        return np.rad2deg(delta_theta)
