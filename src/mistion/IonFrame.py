import itertools
import os
import shutil
import tempfile
from datetime import datetime
from multiprocessing import cpu_count, Pool
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from .DLayer import DLayer
from .FLayer import FLayer
from .modules.helpers import none_or_array, elaz_mesh, TextColor, pic2vid
from .modules.ion_tools import trop_refr
from .modules.parallel import calc_refatt_par, calc_refatt
from .modules.plotting import polar_plot_star, polar_plot


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
    :param _pbar: If True - a progress bar will appear.
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt: datetime,
            position: Tuple[float, float, float],
            nside: int = 128,
            dbot: float = 60,
            dtop: float = 90,
            ndlayers: int = 10,
            fbot: float = 150,
            ftop: float = 500,
            nflayers: int = 30,
            _pbar: bool = False,
            _autocalc: bool = True,
    ):
        if isinstance(dt, datetime):
            self.dt = dt
        else:
            raise ValueError("Parameter dt must be a datetime object.")
        self.position = position
        self.nside = nside
        self.dlayer = DLayer(dt, position, dbot, dtop, ndlayers, nside, _pbar, _autocalc)
        self.flayer = FLayer(dt, position, fbot, ftop, nflayers, nside, _pbar, _autocalc)

    @staticmethod
    def _parallel_calc(func, el, az, freq, pbar_desc, **kwargs):
        """
        Sends methods either to serial or parallel calculation routines based on type of freq.
        """
        if (isinstance(freq, list) or isinstance(freq, np.ndarray)) and len(freq) > 1:
            return calc_refatt_par(func, el, az, freq, pbar_desc, **kwargs)
        else:
            return calc_refatt(func, el, az, freq, **kwargs)

    @staticmethod
    def troprefr(el: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Approximation of the refraction in the troposphere recommended by the ITU-R:
        https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-7-201510-S!!PDF-E.pdf

        :param el: Elevation of observation(s) in [deg].
        :return: Refraction in the troposphere in [deg].
        """
        return np.rad2deg(trop_refr(np.deg2rad(90 - el)))

    def refr(self, el, az, freq, troposphere=True, _pbar_desc=None):
        """
        Calculates refraction in F layer for a given model of ionosphere. Output is the change of zenith angle theta
        (theta -> theta + dtheta). If coordinates are floats the output will be a single number; if they are arrays -
        the output will be a 2D array with dimensions `az.size` x `el.size` (according to `np.meshgrid(el, az)`).

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
        return self._parallel_calc(
            self.flayer.refr, el, az, freq, _pbar_desc, troposphere=troposphere
        )

    def atten(self, el, az, freq, _pbar_desc=None, col_freq="default", troposphere=True):
        """
        Calculates attenuation in D layer for a given model of ionosphere. Output is the attenuation factor between 0
        (total attenuation) and 1 (no attenuation). If coordinates are floats the output will be a single number; if
        they are arrays - the output will be a 2D array with dimensions `el.size` x `az.size`.

        Parameters
        ----------
        el : None | float | np.ndarray
            Elevation of observation(s). If not provided - the model's grid will be used.
        az : None | float | np.ndarray
            Azimuth of observation(s). If not provided - the model's grid will be used.
        col_freq : str, float
            The collision frequency ('default', 'nicolet', 'setty', 'aggrawal', or float in Hz)
        troposphere : Bool, default=True
            Account for troposphere refraction bias

        Returns
        -------
        np.ndarray
        """
        return self._parallel_calc(
            self.dlayer.atten, el, az, freq, _pbar_desc, col_freq=col_freq, troposphere=troposphere
        )

    def write_self_to_file(self, file):
        h5dir = f"{self.dt.year:04d}{self.dt.month:02d}{self.dt.day:02d}{self.dt.hour:02d}{self.dt.minute:02d}"
        grp = file.create_group(h5dir)
        meta = grp.create_dataset("meta", shape=(0,))
        meta.attrs["position"] = self.position
        meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
        meta.attrs["nside"] = self.nside

        meta.attrs["ndlayers"] = self.dlayer.ndlayers
        meta.attrs["dtop"] = self.dlayer.dtop
        meta.attrs["dbot"] = self.dlayer.dbot

        meta.attrs["nflayers"] = self.flayer.nflayers
        meta.attrs["fbot"] = self.flayer.fbot
        meta.attrs["ftop"] = self.flayer.ftop

        if (
                np.average(self.dlayer.d_e_density) > 0
                and np.average(self.flayer.f_e_density) > 0
        ):
            grp.create_dataset("d_e_density", data=self.dlayer.d_e_density)
            grp.create_dataset("d_e_temp", data=self.dlayer.d_e_temp)
            grp.create_dataset("f_e_density", data=self.flayer.f_e_density)
            grp.create_dataset("f_e_temp", data=self.flayer.f_e_temp)

    def save(self, savedir=None, name=None):
        import h5py

        filename = f"{self.dt.year:04d}{self.dt.month:02d}{self.dt.day:02d}{self.dt.hour:02d}{self.dt.minute:02d}"
        savedir = savedir or "ion_models/"
        os.makedirs(savedir, exist_ok=True)

        name = name or filename
        name = os.path.join(savedir, name)
        if not name.endswith(".h5"):
            name += ".h5"

        file = h5py.File(name, mode="w")
        self.write_self_to_file(file)
        file.close()

    @classmethod
    def read_self_from_file(cls, grp):
        meta = grp.get("meta")
        obj = cls(
            _autocalc=False,
            dt=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
            position=meta.attrs["position"],
            nside=meta.attrs["nside"],
            dbot=meta.attrs["dbot"],
            dtop=meta.attrs["dtop"],
            ndlayers=meta.attrs["ndlayers"],
            fbot=meta.attrs["fbot"],
            ftop=meta.attrs["ftop"],
            nflayers=meta.attrs["nflayers"],
        )
        obj.dlayer.d_e_density = none_or_array(grp.get("d_e_density"))
        obj.dlayer.d_e_temp = none_or_array(grp.get("d_e_temp"))

        obj.flayer.f_e_density = none_or_array(grp.get("f_e_density"))
        obj.flayer.f_e_temp = none_or_array(grp.get("f_e_temp"))
        return obj

    @classmethod
    def load(cls, path: str):
        import h5py

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

    def plot_ded(self, gridsize=200, layer=None, **kwargs):
        barlabel = r"$m^{-3}$"
        el, az = elaz_mesh(gridsize)
        ded = self.dlayer.ded(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, ded),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_det(self, gridsize=200, layer=None, **kwargs):
        barlabel = r"$K^\circ$"
        el, az = elaz_mesh(gridsize)
        det = self.dlayer.det(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, det),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_fed(self, gridsize=200, layer=None, **kwargs):
        barlabel = r"$m^{-3}$"
        el, az = elaz_mesh(gridsize)
        fed = self.flayer.fed(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, fed),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_fet(self, gridsize=200, layer=None, **kwargs):
        barlabel = r"$K^\circ$"
        el, az = elaz_mesh(gridsize)
        fet = self.flayer.fet(el, az, layer)
        return polar_plot(
            (np.deg2rad(az), 90 - el, fet),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_atten(self, freq, troposphere=True, gridsize=200, **kwargs):
        el, az = elaz_mesh(gridsize)
        atten = self.dlayer.atten(el, az, freq, troposphere=troposphere)
        barlabel = None
        return polar_plot(
            (np.deg2rad(az), 90 - el, atten),
            dt=self.dt,
            pos=self.position,
            freq=freq,
            barlabel=barlabel,
            **kwargs,
        )

    def plot_refr(self, freq, troposphere=True, gridsize=200, cmap="viridis_r", **kwargs):
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
            **kwargs,
        )

    def plot_troprefr(self, gridsize=200, cmap="viridis_r", **kwargs):
        el, az = elaz_mesh(gridsize)
        troprefr = self.troprefr(el)
        barlabel = r"$deg$"
        return polar_plot(
            (np.deg2rad(az), 90 - el, troprefr),
            dt=self.dt,
            pos=self.position,
            barlabel=barlabel,
            cmap=cmap,
            **kwargs,
        )

    def _freq_animation(
            self,
            func,
            name,
            freqrange=(45e6, 125e6),
            gridsize=100,
            fps=20,
            duration=5,
            savedir="animations/",
            title=None,
            barlabel=None,
            plotlabel=None,
            dpi=300,
            cmap="viridis",
            cbformat="%.2f",
            pbar_label="",
    ):
        print(
            TextColor.BOLD
            + TextColor.YELLOW
            + "Animation making procedure started"
            + f" [{pbar_label}]"
            + TextColor.END
            + TextColor.END
        )
        el, az = elaz_mesh(gridsize)
        nframes = duration * fps
        freqs = np.linspace(*freqrange, nframes)[::-1]
        data = np.array(
            func(el, az, freqs, _pbar_desc="[1/3] Calculating data")
        )
        cbmax = np.nanmax(data)
        cbmin = np.nanmin(data)

        tmpdir = tempfile.mkdtemp()
        nproc = np.min([cpu_count(), len(freqs)])
        plot_data = [(np.deg2rad(az), 90 - el, data[i]) for i in range(len(data))]
        plot_saveto = [os.path.join(tmpdir, str(i).zfill(6)) for i in range(len(data))]
        with Pool(processes=nproc) as pool:
            list(
                tqdm(
                    pool.imap(
                        polar_plot_star,
                        zip(
                            plot_data,
                            itertools.repeat(self.dt),
                            itertools.repeat(self.position),
                            freqs,
                            itertools.repeat(title),
                            itertools.repeat(barlabel),
                            itertools.repeat(plotlabel),
                            itertools.repeat((cbmin, cbmax)),
                            plot_saveto,
                            itertools.repeat(dpi),
                            itertools.repeat(cmap),
                            itertools.repeat(cbformat),
                        ),
                    ),
                    desc="[2/3] Rendering frames",
                    total=len(freqs),
                )
            )
        desc = "[3/3] Rendering video"
        pic2vid(tmpdir, name, fps=fps, desc=desc, savedir=savedir)

        shutil.rmtree(tmpdir)
        return

    def animate_atten_vs_freq(self, name, **kwargs):
        self._freq_animation(
            self.atten,
            name,
            pbar_label="D layer attenuation",
            cbformat="%.3f",
            **kwargs,
        )

    def animate_refr_vs_freq(self, name, cmap="viridis_r", **kwargs):
        self._freq_animation(
            self.refr,
            name,
            pbar_label="F layer refraction",
            barlabel=r"deg",
            cmap=cmap,
            cbformat="%.2f",
            **kwargs,
        )
