from multiprocessing import Pool, cpu_count
import itertools

import iricore
import numpy as np
import pymap3d as pm
import healpy as hp
from tqdm import tqdm

from mistion.modules.helpers import Ellipsoid, eval_layer, iri_star
from mistion.modules.ion_tools import srange, n_f, refr_angle, trop_refr


class FLayer:
    def __init__(
        self,
        dt,
        position,
        fbot=60,
        ftop=90,
        nflayers=30,
        nside=128,
        autocalc: bool = True
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
        if autocalc:
            self._calc_par()

    def _calc(self):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density, electron
        temperature, attenuation factor, average temperature] and F-layer [electron density, angle of the outgoing
        refracted beam at each layer, the net deviation of the elevation angle for each coordinate, refractive index
        at each layer].
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
        nproc = np.min([cpu_count(), self.nflayers])
        heights = [(h, h, 1) for h in np.linspace(self.fbot, self.ftop, self.nflayers)]
        with Pool(processes=nproc) as pool:
            res = list(
                tqdm(
                    pool.imap(
                        iri_star,
                        zip(
                            itertools.repeat(self.dt),
                            heights,
                            itertools.repeat(self._obs_lats),
                            itertools.repeat(self._obs_lons),
                            itertools.repeat(0.),
                        ),
                    ),
                    total=self.nflayers,
                    disable=not pbar,
                    desc="F layer",
                )
            )
            self.f_e_density = np.vstack([r["ne"][:, 0] for r in res]).T
            self.f_e_temp = np.vstack([r["te"][:, 0] for r in res]).T
        return

    def fed(self, el, az, layer=None):
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

    def frefr(self, el, az, freq, troposphere=True):
        """
        #TODO
        """
        R_E = 6378100.0
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
        d_tel = R_E + self.position[2]  # Distance from Earth center to telescope
        d_cur = R_E + f_heights[0]  # Distance from Earth center to layer

        # The inclination angle at the 1st interface using law of cosines [rad]
        costheta_inc = (r_slant**2 + d_cur**2 - d_tel**2) / (2 * r_slant * d_cur)
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
            d_next = R_E + h_next

            # Angle between d_cur and r_slant
            int_angle = np.pi - theta_ref
            # The inclination angle at the i-th interface using law of sines [rad]
            theta_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

            # Getting r2 using law of cosines
            r_slant = srange(np.deg2rad(90 - el_cur), d_next - d_cur, R_E=R_E + d_cur)
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
