from __future__ import annotations

import itertools
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Union, Sequence, Literal, Tuple

import h5py
import numpy as np

from .IonLayer import IonLayer
from .modules.helpers import none_or_array, altaz_mesh, open_save_file
from .modules.ion_tools import trop_refr
from .modules.parallel import shared_array
from .modules.plotting import polar_plot
from .raytracing import raytrace_star


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
            nside: int = 64,
            hbot: float = 60,
            htop: float = 500,
            nlayers: int = 500,
            rdeg_offset: float = 5,
            iriversion: Literal[16, 20] = 20,
            echaim: bool = False,
            autocalc: bool = True,
            _pool: Union[Pool, None] = None,
            **kwargs,
    ):
        if isinstance(dt, datetime):
            self.dt = dt
        else:
            raise ValueError("Parameter dt must be a datetime object.")
        self.position = position
        self.nside = nside
        self.rdeg_offset = rdeg_offset
        self.iriversion = iriversion
        self.echaim = echaim
        self.layer = IonLayer(
            dt=dt,
            position=position,
            hbot=hbot,
            htop=htop,
            nlayers=nlayers,
            rdeg_offset=rdeg_offset,
            nside=nside,
            name='Calculating Ne and Te',
            iriversion=iriversion,
            autocalc=autocalc,
            echaim=echaim,
            _pool=_pool,
            **kwargs,
        )

    def __call__(self,
                 alt: float | np.ndarray,
                 az: float | np.ndarray,
                 freq: float,
                 col_freq: str = "default",
                 troposphere: bool = True,
                 height_profile: bool = False,
                 _pool: Union[Pool, None] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        b_alt = np.atleast_1d(alt).astype(np.float64)
        b_az = np.atleast_1d(az).astype(np.float64)
        nproc = np.min([len(b_alt), cpu_count()])
        b_alt = np.array_split(b_alt, nproc)
        b_az = np.array_split(b_az, nproc)

        pool = Pool(processes=nproc) if _pool is None else _pool

        sh_edens = shared_array(self.layer.edens)
        sh_etemp = shared_array(self.layer.etemp)
        init_dict = self.layer.get_init_dict()

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
            f"\tBottom height:\t{self.layer.hbot} [km]\n"
            f"\tTop height:\t{self.layer.htop} [km]\n"
            f"\tN sublayers:\t{self.layer.nlayers}\n"
        )

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

        if self.layer is not None:
            meta.attrs["rdeg_offset"] = self.layer.rdeg_offset
            meta.attrs["nlayers"] = self.layer.nlayers
            meta.attrs["htop"] = self.layer.htop
            meta.attrs["hbot"] = self.layer.hbot
            grp.create_dataset("edens", data=self.layer.edens)
            grp.create_dataset("etemp", data=self.layer.etemp)

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
        obj.layer.edens = none_or_array(grp.get("edens"))
        obj.layer.etemp = none_or_array(grp.get("etemp"))
        if obj.layer.edens is None and obj.layer.etemp is None:
            obj.layer = None
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
        edens = self.layer.ed(alt, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - alt, edens),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
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
        fet = self.layer.et(alt, az, layer)
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

    def calc(self, pbar: bool = False):
        """
        Calculates the layer's electron density and temperatur (use it if you set autocalc=False during the initialization).

        :param pbar: If True - a progress bar will appear.
        """
        self.layer.calc(pbar)

    def troprefr(self, alt: float | np.ndarray) -> float | np.ndarray:
        """
        Approximation of the refraction in the troposphere recommended by the ITU-R:
        https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-9-201712-I!!PDF-E.pdf

        :param alt: Elevation of observation(s) in [deg].
        :return: Refraction in the troposphere in [deg].
        """
        return trop_refr(alt, self.position[2] * 1e-3)

    def get_init_dict(self):
        """
        Returns a dictionary containing the initial parameters for the IonFrame object.

        Note:
            - The default value for autocalc is False.
        """
        return {
            'dt': self.dt,
            'position': self.position,
            'nside': self.nside,
            'hbot': self.layer.hbot,
            'htop': self.layer.htop,
            'nlayers': self.layer.nlayers,
            'rdeg_offset': self.rdeg_offset,
            'iriversion': self.iriversion,
            'echaim': self.echaim,
        }
