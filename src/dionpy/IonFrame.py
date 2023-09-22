from __future__ import annotations

import os
from datetime import datetime
from multiprocessing import Pool
from typing import Union, Sequence, Literal

import h5py
import numpy as np

from .IonLayer import IonLayer
from .modules.helpers import none_or_array, elaz_mesh
from .modules.ion_tools import trop_refr
from .modules.parallel import shared_array
from .modules.plotting import polar_plot

from .raytracing import raytrace

# TODO: add height constraints in plotting
# TODO: rewrite saving


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
    :param dbot: Lower limit in [km] of the D layer of the ionosphere.
    :param dtop: Upper limit in [km] of the D layer of the ionosphere.
    :param ndlayers: Number of sub-layers in the D layer for intermediate calculations.
    :param fbot: Lower limit in [km] of the F layer of the ionosphere.
    :param ftop: Upper limit in [km] of the F layer of the ionosphere.
    :param nflayers: Number of sub-layers in the F layer for intermediate calculations.
    :param iriversion: Version of the IRI model to use. Must be a two digit integer that refers to
                        the last two digits of the IRI version number. For example, version 20 refers
                        to IRI-2020.
    :param echaim: Use ECHAIM model for electron density estimation.
    :param autocalc: If True - the model will be calculated immediately after definition.
    :param single_layer: If selected - only one layer will be calculated.
    :param _pbar: If True - a progress bar will appear.
    :param **kwargs: Other parameters for ionospheric layer initialization.
    """

    def __init__(
            self,
            dt: datetime,
            position: Sequence[float, float, float],
            nside: int = 64,
            hbot: float = 60,
            htop: float = 500,
            nlayers: int = 300,
            iriversion: Literal[16, 20] = 20,
            echaim: bool = False,
            autocalc: bool = True,
            _pbar: bool = False,
            _pool: Union[Pool, None] = None,
            **kwargs
    ):
        if isinstance(dt, datetime):
            self.dt = dt
        else:
            raise ValueError("Parameter dt must be a datetime object.")
        self.position = position
        self.nside = nside
        self.iriversion = iriversion
        self.layer = IonLayer(
            dt=dt,
            position=position,
            hbot=hbot,
            htop=htop,
            nlayers=nlayers,
            nside=nside,
            pbar=_pbar,
            name='Calculating Ne and Te',
            iriversion=iriversion,
            autocalc=autocalc,
            echaim=echaim,
            _pool=_pool,
            **kwargs,
        )

    def calc_layer(self, pbar: bool = False):
        """
        Calculates the layer's edens and etemp (use it if you set autocalc=False during the initialization).

        :param pbar: If True - a progress bar will appear.
        """
        self.layer.calc(pbar)

    def __call__(self,
                 alt: float | np.ndarray,
                 az: float | np.ndarray,
                 freq: float | np.ndarray,
                 _pbar_desc: str | None = None,
                 col_freq: str = "default",
                 troposphere: bool = True,
                 height_profile: bool = False) -> float | np.ndarray:

        sh_edens = shared_array(self.layer.edens)
        sh_etemp = shared_array(self.layer.etemp)
        init_dict = self.layer.get_init_dict()
        dtheta = raytrace(init_dict, sh_edens, sh_etemp, alt, az, freq)
        print(dtheta)
        return dtheta

    def troprefr(self, alt: float | np.ndarray) -> float | np.ndarray:
        """
        Approximation of the refraction in the troposphere recommended by the ITU-R:
        https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-9-201712-I!!PDF-E.pdf

        :param alt: Elevation of observation(s) in [deg].
        :return: Refraction in the troposphere in [deg].
        """
        return trop_refr(alt, self.position[2] * 1e-3)

    def radec2altaz(self, ra: float | np.ndarray, dec: float | np.ndarray):
        """
        Converts sky coordinates to altitude and azimuth angles in horizontal CS.

        :param ra: Right ascension in [deg].
        :param dec: Declination in [deg].
        :return: [alt, az], both in [deg]
        """
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz
        from astropy.time import Time
        from astropy import units as u

        location = EarthLocation(lat=self.position[0], lon=self.position[1], height=self.position[2] * u.m)
        time = Time(self.dt)
        altaz_cs = AltAz(location=location, obstime=time)
        skycoord = SkyCoord(ra * u.deg, dec * u.deg)
        aa_coord = skycoord.transform_to(altaz_cs)
        return aa_coord.alt.value, aa_coord.az.value

    def write_self_to_file(self, file: h5py.File):
        h5dir = f"{self.dt.year:04d}{self.dt.month:02d}{self.dt.day:02d}{self.dt.hour:02d}{self.dt.minute:02d}"
        grp = file.create_group(h5dir)
        meta = grp.create_dataset("meta", shape=(0,))
        meta.attrs["position"] = self.position
        meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
        meta.attrs["nside"] = self.nside
        meta.attrs["iriversion"] = self.iriversion

        if self.dlayer is not None:
            meta.attrs["ndlayers"] = self.dlayer.nlayers
            meta.attrs["dtop"] = self.dlayer.htop
            meta.attrs["dbot"] = self.dlayer.hbot
            grp.create_dataset("dedens", data=self.dlayer.edens)
            grp.create_dataset("detemp", data=self.dlayer.etemp)
        if self.flayer is not None:
            meta.attrs["nflayers"] = self.flayer.nlayers
            meta.attrs["fbot"] = self.flayer.hbot
            meta.attrs["ftop"] = self.flayer.htop
            grp.create_dataset("fedens", data=self.flayer.edens)
            grp.create_dataset("fetemp", data=self.flayer.etemp)

    def save(self, saveto: str = "./ionframe"):
        """
        Save the model to HDF file.

        :param saveto: Path and name of the file.
        """
        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)
        if not saveto.endswith(".h5"):
            saveto += ".h5"

        file = h5py.File(saveto, mode="w")
        self.write_self_to_file(file)
        file.close()

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
        obj.dlayer.edens = none_or_array(grp.get("dedens"))
        obj.dlayer.etemp = none_or_array(grp.get("detemp"))
        if obj.dlayer.edens is None and obj.dlayer.etemp is None:
            obj.dlayer = None

        obj.flayer.edens = none_or_array(grp.get("fedens"))
        obj.flayer.etemp = none_or_array(grp.get("fetemp"))
        if obj.flayer.edens is None and obj.flayer.etemp is None:
            obj.flayer = None
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
        el, az = elaz_mesh(gridsize)
        fed = self.layer.ed(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, fed),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            **kwargs,
        )

    def plot_et(self, gridsize: int = 200, layer: int | None = None, cmap='viridis', **kwargs):
        """
        Visualize electron temperature in the ionospheric layer.

        :param gridsize: Grid resolution of the plot.
        :param layer: A specific sub-layer to plot. If None - an average of all layers is calculated.
        :param cmap: A colormap to use in the plot.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        barlabel = r"K"
        el, az = elaz_mesh(gridsize)
        fet = self.layer.et(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, fet),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            **kwargs,
        )

    def plot_atten(
            self, freq: float, troposphere: bool = True, gridsize: int = 200, cmap='Purples_r', cblim=None, **kwargs
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
        el, az = elaz_mesh(gridsize)
        atten = self.dlayer.atten(el, az, freq, troposphere=troposphere)
        cblim = cblim or [None, 1]
        # atten_db = 20 * np.log10(atten)
        # barlabel = r"dB"
        return polar_plot(
            (np.deg2rad(az), 90 - el, atten),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )

    def plot_emiss(
            self, freq: float, troposphere: bool = True, gridsize: int = 200, cmap='Oranges', cblim=None, **kwargs
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
        el, az = elaz_mesh(gridsize)
        _, emiss = self.dlayer.atten(el, az, freq, troposphere=troposphere, emission=True)
        cblim = cblim or [0, None]
        barlabel = r"$K$"
        return polar_plot(
            (np.deg2rad(az), 90 - el, emiss),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            barlabel=barlabel,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )

    def plot_refr(
            self,
            freq: float,
            troposphere: bool = True,
            gridsize: int = 200,
            cmap: str = 'Greens',
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
        el, az = elaz_mesh(gridsize)
        refr = self.flayer.refr(el, az, freq, troposphere=troposphere)
        barlabel = r"$deg$"
        return polar_plot(
            (np.deg2rad(az), 90 - el, refr),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            barlabel=barlabel,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )

    def plot_troprefr(self, gridsize=200, cmap="Greens", cblim=None, **kwargs):
        """
        Visualize tropospheric refraction.

        :param gridsize: Grid resolution of the plot.
        :param cmap: A colormap to use in the plot.
        :param cblim: Colorbar limits.
        :param kwargs: See `dionpy.plot_kwargs`.
        :return: A matplotlib figure.
        """
        el, az = elaz_mesh(gridsize)
        troprefr = self.troprefr(el)
        cblim = cblim or [0, None]
        barlabel = r"$deg$"
        return polar_plot(
            (np.deg2rad(az), 90 - el, troprefr),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            cblim=cblim,
            **kwargs,
        )
