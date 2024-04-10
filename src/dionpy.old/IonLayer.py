from __future__ import annotations

import ctypes
import itertools
from datetime import datetime
import multiprocessing as mp
from typing import Union, Sequence

import healpy as hp
import numpy as np
from tqdm import tqdm

from .modules.helpers import eval_layer, R_EARTH
from .modules.parallel import create_shared_block
from .modules.parallel_iri import parallel_iri, parallel_echaim


def _estimate_ahd(htop: float, hint: float = 0, r: float = R_EARTH*1e-3):
    """
    Estimates the angular horizontal distance (ahd) between the top point of an atmospheric
    layer and the Earth's surface.

    :param htop: The height of the top point of the atmospheric layer in [km].
    :param hint: The height above the Earth's surface in [km].
    :param r: The radius of the Earth in [km].
    """
    return np.rad2deg(np.arccos(r / (r + hint)) + np.arccos(r / (r + htop)))


class IonLayer:
    """
    A model of a layer of specific height range in the ionosphere. Includes electron density and temperature data after
    calculation.

    :param dt: Date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param hbot: Lower limit in [km] of the layer of the ionosphere.
    :param htop: Upper limit in [km] of the layer of the ionosphere.
    :param nlayers: Number of sub-layers in the layer for intermediate calculations.
    :param nside: Resolution of healpix grid.
    :param rdeg_offset: Extend radius of coordinate plane in [deg].
    :param name: Name of the layer for description use.
    :param iriversion: Version of the IRI model to use. Must be a two digit integer that refers to
                        the last two digits of the IRI version number. For example, version 20 refers
                        to IRI-2020.
    :param autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt: datetime,
            position: Sequence[float, float, float],
            hbot: float,
            htop: float,
            nlayers: int = 100,
            nside: int = 64,
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
                raise ValueError(
                    "The E-CHAIM model does not cover all coordinates needed for the ionosphere model at the "
                    "specified instrument's position.")

        self.hbot = hbot
        self.htop = htop
        self.nlayers = nlayers
        self.dt = dt
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
            import time
            t1 = time.time()
            self.calc(_pool=_pool)
            # print(self.edens.mean())
            print(time.time() - t1)

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
            autocalc=False,
            echaim=self.echaim,
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
        for i in range(nbatches-1):
            batch_i[i+1] = batch_i[i] + len(batch_lat[i])
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
        print(np.average(self.edens))
        print(np.average(shedens))
        self.edens[:] = shedens[:]
        shm_edens.close()
        shm_edens.unlink()
        assert 1==0

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
