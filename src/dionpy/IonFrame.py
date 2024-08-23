from __future__ import annotations
from __future__ import annotations

import itertools
import warnings
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple
from typing import Union, Sequence

import h5py
import healpy as hp
import iricore.iri
from iricore.iri import indices_uptodate
import numpy as np

from .modules.helpers import eval_layer, R_EARTH
from .modules.helpers import none_or_array, altaz_mesh, open_save_file
from .modules.ion_tools import trop_refr, plasfreq
from .modules.parallel import create_shared_block
from .modules.parallel import shared_array
from .modules.parallel_iri import parallel_iri, parallel_echaim
from .modules.plotting import polar_plot


def _estimate_ahd(htop: float, hint: float = 0, r: float = R_EARTH * 1e-3):
    """
    Estimates the angular horizontal distance (ahd) between the top point of an atmospheric
    layer and the Earth's surface.

    :param htop: The height of the top point of the atmospheric layer in [km].
    :param hint: The height above the Earth's surface in [km].
    :param r: The radius of the Earth in [km].
    """
    return np.rad2deg(np.arccos(r / (r + hint)) + np.arccos(r / (r + htop)))


class IonFrame:
    """
    A model of the ionosphere for a specific moment in time. Given a position, calculates electron
    density and temperature in the ionosphere in all visible directions using International Reference
    Ionosphere (IRI) model. The calculated model can estimate ionospheric attenuation and refraction
    in a given direction defined by elevation and azimuth angles.

    :param dt: Date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param nside: Resolution of healpix grid.
    :param hbot: Lower limit in [km] of the layer of the ionosphere.
    :param htop: Upper limit in [km] of the layer of the ionosphere.
    :param nlayers: Number of sub-layers in the ionospheric layer for intermediate calculations.
    :param rdeg_offset: Extends the angular horizon distance of calculated ionosphere in [degrees].
    :param iriversion: Version of the IRI model to use. Must be a two digit integer that refers to
                        the last two digits of the IRI version number. For example, version 20 refers
                        to IRI-2020.
    :param echaim: Use ECHAIM model for electron density estimation.
    :param autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt: datetime,
            position: Sequence[float, float, float],
            hbot: float = 60,
            htop: float = 1000,
            nlayers: int = 100,
            nside: int = 32,
            rdeg_offset: float = 5,
            name: str | None = None,
            iriversion: int = 20,
            autocalc: bool = True,
            echaim: bool = False,
            _pool: Union[mp.Pool, None] = None,
    ):
        self.rdeg = _estimate_ahd(htop, position[-1] * 1e-3) + rdeg_offset
        self.rdeg_offset = rdeg_offset

        if echaim:
            if position[0] - self.rdeg < 55:
                warnings.warn("ECHAIM does not cover the whole field of view for you position. "
                              "All electron density below 55 deg latitude will be NaN. This is important "
                              "because these NaNs can be confused with wave absorption when raytracing.", stacklevel=2)

        self.hbot = hbot
        self.htop = htop
        self.nlayers = nlayers
        if isinstance(dt, datetime):
            self.dt = dt
        else:
            raise ValueError("Parameter dt must be a datetime object.")

        indices_uptodate(dt)

        self.position = position
        self.name = name
        self.echaim = echaim

        self.nside = nside
        self.iriversion = iriversion
        self._posvec = hp.ang2vec(self.position[1], self.position[0], lonlat=True)
        self._obs_pixels = hp.query_disc(
            self.nside, self._posvec, np.deg2rad(self.rdeg), inclusive=True
        )
        self._obs_lons, self._obs_lats = hp.pix2ang(
            self.nside, self._obs_pixels, lonlat=True
        )
        self.edens = np.zeros((len(self._obs_pixels), nlayers), dtype=np.float32)
        self.etemp = np.zeros((len(self._obs_pixels), nlayers), dtype=np.float32)

        if autocalc:
            self.calc(_pool=_pool)

    def __call__(self,
                 alt: float | np.ndarray,
                 az: float | np.ndarray,
                 freq: float,
                 col_freq: str = "default",
                 troposphere: bool = True,
                 height_profile: bool = False,
                 _pool: Union[Pool, None] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from .raytracing import raytrace_star
        b_alt = np.atleast_1d(alt).astype(np.float64)
        b_az = np.atleast_1d(az).astype(np.float64)
        nproc = np.min([len(b_alt), cpu_count()])
        b_alt = np.array_split(b_alt, nproc)
        b_az = np.array_split(b_az, nproc)

        # pool = Pool(processes=nproc) if _pool is None else _pool
        pool = _pool or mp.get_context('fork').Pool(processes=nproc)

        sh_edens = shared_array(self.edens)
        sh_etemp = shared_array(self.etemp)
        init_dict = self.get_init_dict()

        res = list(
            pool.imap(
                raytrace_star,
                zip(
                    itertools.repeat(init_dict),
                    itertools.repeat(sh_edens),
                    itertools.repeat(sh_etemp),
                    b_alt,
                    b_az,
                    itertools.repeat(freq),
                    itertools.repeat(col_freq),
                    itertools.repeat(troposphere),
                    itertools.repeat(height_profile),
                ),
            )
        )
        dtheta = np.squeeze(np.concatenate([x[0] for x in res], axis=0))
        atten = np.squeeze(np.concatenate([x[1] for x in res], axis=0))
        emiss = np.squeeze(np.concatenate([x[2] for x in res], axis=0))
        return dtheta, atten, emiss

    def __str__(self):
        return (
            f"IonFrame instance\n"
            f"Date:\t{self.dt.strftime('%d %b %Y %H:%M:%S')} UTC\n"
            f"Position:\n"
            f"\tlat = {self.position[0]:.2f} [deg]\n"
            f"\tlon = {self.position[1]:.2f} [deg]\n"
            f"\talt = {self.position[2]:.2f} [m]\n"
            f"NSIDE:\t{self.nside}\n"
            f"IRI version:\t20{self.iriversion}\n"
            f"Use E-CHAIM:\t{self.echaim}\n"
            f"Layer properties:\n"
            f"\tBottom height:\t{self.hbot} [km]\n"
            f"\tTop height:\t{self.htop} [km]\n"
            f"\tN sublayers:\t{self.nlayers}\n"
        )

    def get_init_dict(self):
        """
        Returns a dictionary containing the initial parameters for the IonLayer object.

        Note:
            - The default value for autocalc is False.
        """
        return dict(
            dt=self.dt,
            position=self.position,
            hbot=self.hbot,
            htop=self.htop,
            nlayers=self.nlayers,
            nside=self.nside,
            rdeg_offset=self.rdeg_offset,
            iriversion=self.iriversion,
            echaim=self.echaim,
            autocalc=False,
        )

    def _batch_split(self, batch):
        nbatches = len(self._obs_pixels) // batch + 1
        nproc = np.min([mp.cpu_count(), nbatches])
        blat = np.array_split(self._obs_lats, nbatches)
        blon = np.array_split(self._obs_lons, nbatches)
        return nbatches, nproc, blat, blon

    def calc(self, _pool=None):
        heights = (
            self.hbot,
            self.htop,
            (self.htop - self.hbot) / (self.nlayers - 1) - 1e-6,
        )

        batch_size = 200
        nbatches, nproc, batch_lat, batch_lon = self._batch_split(batch_size)
        batch_i = np.zeros(nbatches, dtype=np.int32)
        for i in range(nbatches - 1):
            batch_i[i + 1] = batch_i[i] + len(batch_lat[i])
        shm_edens, shedens = create_shared_block(self.edens)
        shm_etemp, shetemp = create_shared_block(self.etemp)

        pool = _pool or mp.get_context('fork').Pool(processes=nproc)
        pool.starmap(
            parallel_iri,
            zip(
                itertools.repeat(self.dt),
                itertools.repeat(heights),
                batch_lat,
                batch_lon,
                itertools.repeat(shm_edens.name),
                itertools.repeat(shm_etemp.name),
                itertools.repeat(self.edens.shape),
                batch_i,
                itertools.repeat(self.iriversion),
            )
        )

        if _pool is None:
            pool.close()

        self.edens[:] = shedens[:]
        self.etemp[:] = shetemp[:]

        shm_edens.close()
        shm_edens.unlink()
        shm_etemp.close()
        shm_etemp.unlink()

        if self.echaim:
            self._calc_echaim(_pool=_pool)

    def _calc_echaim(self, _pool: Union[mp.Pool, None] = None):
        """
        Replace electron density with that calculated with ECHAIM.
        """
        heights = np.linspace(self.hbot, self.htop, self.nlayers, endpoint=True)
        batch_size = 100
        nbatches, nproc, batch_lat, batch_lon = self._batch_split(batch_size)

        batch_i = np.zeros(nbatches, dtype=np.int32)
        for i in range(nbatches - 1):
            batch_i[i + 1] = batch_i[i] + len(batch_lat[i])
        shm_edens, shedens = create_shared_block(self.edens)

        pool = _pool or mp.get_context('fork').Pool(processes=nproc)
        pool.starmap(
            parallel_echaim,
            zip(
                batch_lat,
                batch_lon,
                itertools.repeat(heights),
                itertools.repeat(self.dt),
                itertools.repeat(shm_edens.name),
                itertools.repeat(self.edens.shape),
                batch_i,
                itertools.repeat(True),
                itertools.repeat(True),
                itertools.repeat(True),
            )
        )

        if _pool is None:
            pool.close()
        self.edens[:] = shedens[:]

    def ed(
            self,
            alt: float | np.ndarray,
            az: float | np.ndarray,
            layer: int | None = None,
    ) -> float | np.ndarray:
        """
        :param alt: Elevation of an observation.
        :param az: Azimuth of an observation.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron density in the layer.
        """
        return eval_layer(
            alt,
            az,
            self.nside,
            self.position,
            self.hbot,
            self.htop,
            self.nlayers,
            self._obs_pixels,
            self.edens,
            layer=layer,
        )

    def plasfreq(
            self,
            alt: float | np.ndarray,
            az: float | np.ndarray,
            layer: int | None = None,
            angular: bool = True,
    ) -> float | np.ndarray:
        """
        :param alt: Elevation of an observation.
        :param az: Azimuth of an observation.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :param angular: If True - angular plasma frequency is calculated.
        :return: Plasma frequency in [Hz].
        """
        return plasfreq(self.ed(alt, az, layer), angular=angular)

    def edll(
            self,
            lat: float | np.ndarray,
            lon: float | np.ndarray,
            layer: int | None = None,
    ) -> float | np.ndarray:
        """
        :param lat: Latitude of a point.
        :param lon: Longitude of a point.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron density in the layer.
        """
        map_ = np.zeros(hp.nside2npix(self.nside)) + hp.UNSEEN
        map_[self._obs_pixels] = self.edens[:, layer]
        return hp.pixelfunc.get_interp_val(map_, lon, lat, lonlat=True)

    def et(
            self,
            alt: float | np.ndarray,
            az: float | np.ndarray,
            layer: int | None = None,
    ) -> float | np.ndarray:
        """
        :param alt: Elevation of an observation.
        :param az: Azimuth of an observation.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron temperature in the layer.
        """
        return eval_layer(
            alt,
            az,
            self.nside,
            self.position,
            self.hbot,
            self.htop,
            self.nlayers,
            self._obs_pixels,
            self.etemp,
            layer=layer,
        )

    def etll(
            self,
            lat: float | np.ndarray,
            lon: float | np.ndarray,
            layer: int | None = None,
    ) -> float | np.ndarray:
        """
        :param lat: Latitude of a point.
        :param lon: Longitude of a point.
        :param layer: Number of sublayer from the precalculated sublayers.
                      If None - an average over all layers is returned.
        :return: Electron density in the layer.
        """
        map_ = np.zeros(hp.nside2npix(self.nside)) + hp.UNSEEN
        map_[self._obs_pixels] = self.etemp[:, layer]
        return hp.pixelfunc.get_interp_val(map_, lon, lat, lonlat=True)

    def get_heights(self):
        return np.linspace(self.hbot, self.htop, self.nlayers)

    def raytrace(self,
                 alt: float | np.ndarray,
                 az: float | np.ndarray,
                 freq: float,
                 col_freq: str = "default",
                 troposphere: bool = True,
                 height_profile: bool = False,
                 _pool: Union[Pool, None] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Starts the raytracing procedure and calculates integrated refraction, absorption and emission in all specified
        directions. As a shortcut to this method one can call the IonFrame(...) directly.

        :param alt: Altitude (elevation) of observation in [deg].
        :param az: Azimuth of observation in [deg].
        :param freq: Frequency of observation in [MHz].
        :param col_freq: Model of colission frequency. Available options: \n
                         "default" == "aggrawal" \n
                         "aggrawal": https://ui.adsabs.harvard.edu/abs/1979P%26SS...27..753A/abstract \n
                         "nicolet": [Nicolet, M. 1953, JATP, 3, 200] \n
                         "setty": [Setty, C. S. G. K. 1972, IJRSP, 1, 38]
        :param troposphere: Where to include the tropospheric refraction effect.
        :param height_profile: If True, returns arrays of attenuation and emission before integration and a cumulative
                               history of refraction.
        :returns: (refraction, attenuation, emission)
        """
        return self.__call__(alt, az, freq, col_freq, troposphere, height_profile, _pool)

    # def radec2altaz(self, ra: float | np.ndarray, dec: float | np.ndarray):
    #     """
    #     Converts sky coordinates to altitude and azimuth angles in horizontal CS.
    #
    #     :param ra: Right ascension in [deg].
    #     :param dec: Declination in [deg].
    #     :return: [alt, az], both in [deg]
    #     """
    #     # TODO: make a function outside class
    #     from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    #     from astropy.time import Time
    #     from astropy import units as u
    #
    #     location = EarthLocation(lat=self.position[0], lon=self.position[1], height=self.position[2] * u.m)
    #     time = Time(self.dt)
    #     altaz_cs = AltAz(location=location, obstime=time)
    #     skycoord = SkyCoord(ra * u.deg, dec * u.deg)
    #     aa_coord = skycoord.transform_to(altaz_cs)
    #     return aa_coord.alt.value, aa_coord.az.value

    def write_self_to_file(self, file: h5py.File):
        h5dir = f"{self.dt.year:04d}{self.dt.month:02d}{self.dt.day:02d}{self.dt.hour:02d}{self.dt.minute:02d}"
        grp = file.create_group(h5dir)
        meta = grp.create_dataset("meta", shape=(0,))
        meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
        meta.attrs["position"] = self.position
        meta.attrs["nside"] = self.nside
        meta.attrs["iriversion"] = self.iriversion
        meta.attrs["echaim"] = self.echaim

        meta.attrs["rdeg_offset"] = self.rdeg_offset
        meta.attrs["nlayers"] = self.nlayers
        meta.attrs["htop"] = self.htop
        meta.attrs["hbot"] = self.hbot
        grp.create_dataset("edens", data=self.edens)
        grp.create_dataset("etemp", data=self.etemp)

    def save(self, saveto: str = "./ionframe"):
        """
        Save the model to HDF file.

        :param saveto: Path and name of the file.
        """
        with open_save_file(saveto) as file:
            self.write_self_to_file(file)

    @classmethod
    def read_self_from_file(cls, grp: h5py.Group):
        meta = grp.get("meta")
        meta_attrs = dict(meta.attrs)
        del meta_attrs['dt']

        obj = cls(
            autocalc=False,
            dt=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
            **meta_attrs
        )
        obj.edens = none_or_array(grp.get("edens"))
        obj.etemp = none_or_array(grp.get("etemp"))
        return obj

    @classmethod
    def load(cls, path: str):
        """
        Load a model from file.

        :param path: Path to a file (file extension is not required).
        :return: :class:`IonModel` recovered from a file.
        """
        if not path.endswith(".h5"):
            path += ".h5"
        with h5py.File(path, mode="r") as file:
            groups = list(file.keys())
            if len(groups) > 1:
                raise RuntimeError(
                    "File contains more than one model. "
                    + "Consider reading it with IonModel class."
                )

            grp = file[groups[0]]
            obj = cls.read_self_from_file(grp)
        return obj

    def plot_ed(self, gridsize: int = 200, layer: int | None = None, cmap='plasma', **kwargs):
        """
        Visualize electron density in the ionospheric layer.

        :param gridsize: Grid resolution of the plot.
        :param layer: A specific layer to plot. If None - an average of all layers is calculated.
        :param cmap: A colormap to use in the plot.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        barlabel = r"$m^{-3}$"
        alt, az = altaz_mesh(gridsize)
        edens = self.ed(alt, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - alt, edens),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            **kwargs,
        )

    def plot_plasfreq(self, layer: int, gridsize: int = 200, cmap='plasma', **kwargs):
        """
        Visualize plasma frequency in the ionospheric layer.

        :param gridsize: Grid resolution of the plot.
        :param layer: A specific layer to plot. If None - an average of all layers is calculated.
        :param cmap: A colormap to use in the plot.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        barlabel = r"$MHz$"
        alt, az = altaz_mesh(gridsize)
        data = self.plasfreq(alt, az, layer) * 1e-6
        height = self.get_heights()[layer]
        return polar_plot(
            (np.deg2rad(az), 90 - alt, data),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            height=height,
            **kwargs,
        )

    def plot_et(self, gridsize: int = 200, layer: int | None = None, **kwargs):
        """
        Visualize electron temperature in the ionospheric layer.

        :param gridsize: Grid resolution of the plot.
        :param layer: A specific sub-layer to plot. If None - an average of all layers is calculated.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        barlabel = r"K"
        alt, az = altaz_mesh(gridsize)
        fet = self.et(alt, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - alt, fet),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_atten(
            self, freq: float, troposphere: bool = True, gridsize: int = 200, cmap='plasma', cblim=None, **kwargs
    ):
        """
        Visualize ionospheric attenuation.

        :param freq: Frequency of observation in [Hz].
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param gridsize: Grid resolution of the plot.
        :param cmap: A colormap to use in the plot.
        :param cblim: Colorbar limits.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        alt, az = altaz_mesh(gridsize)
        _, atten, _ = self(alt, az, freq, troposphere=troposphere)
        cblim = cblim or [None, 1]
        return polar_plot(
            (np.deg2rad(az), 90 - alt, atten),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )

    def plot_emiss(
            self, freq: float, troposphere: bool = True, gridsize: int = 200, cblim=None, **kwargs
    ):
        """
        Visualize ionospheric emission.

        :param freq: Frequency of observation in [Hz].
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param gridsize: Grid resolution of the plot.
        :param cblim: Colorbar limits.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        alt, az = altaz_mesh(gridsize)
        _, _, emiss = self(alt, az, freq, troposphere=troposphere)
        cblim = cblim or [0, None]
        barlabel = r"$K$"
        return polar_plot(
            (np.deg2rad(az), 90 - alt, emiss),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            barlabel=barlabel,
            cblim=cblim,
            **kwargs,
        )

    def plot_refr(
            self,
            freq: float,
            troposphere: bool = True,
            gridsize: int = 200,
            cmap: str = "plasma_r",
            cblim=None,
            **kwargs,
    ):
        """
        Visualize ionospheric refraction.

        :param freq: Frequency of observation in [Hz].
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param gridsize: Grid resolution of the plot.
        :param cmap: A colormap to use in the plot.
        :param cblim: Colorbar limits.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        cblim = cblim or [0, None]
        alt, az = altaz_mesh(gridsize)
        refr, _, _ = self(alt, az, freq, troposphere=troposphere)
        barlabel = r"$deg$"
        return polar_plot(
            (np.deg2rad(az), 90 - alt, refr),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            barlabel=barlabel,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )

    def plot_troprefr(self, gridsize=200, cblim=None, **kwargs):
        """
        Visualize tropospheric refraction.

        :param gridsize: Grid resolution of the plot.
        :param cblim: Colorbar limits.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        alt, az = altaz_mesh(gridsize)
        troprefr = self.troprefr(alt)
        cblim = cblim or [0, None]
        barlabel = r"$deg$"
        return polar_plot(
            (np.deg2rad(az), 90 - alt, troprefr),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cblim=cblim,
            **kwargs,
        )

    def troprefr(self, alt: float | np.ndarray) -> float | np.ndarray:
        """
        Approximation of the refraction in the troposphere recommended by the ITU-R:
        https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-9-201712-I!!PDF-E.pdf

        :param alt: Elevation of observation(s) in [deg].
        :return: Refraction in the troposphere in [deg].
        """
        return trop_refr(alt, self.position[2] * 1e-3)
