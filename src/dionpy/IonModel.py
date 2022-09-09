from __future__ import annotations

import itertools
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Callable

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from .IonFrame import IonFrame
from .modules.helpers import elaz_mesh, TextColor, pic2vid, is_iterable
from .modules.parallel import calc_interp_val_par, calc_interp_val, interp_val
from .modules.plotting import polar_plot_star


class IonModel:
    """
    A dynamic model of the ionosphere. Uses a sequence of :class:`IonFrame` objects to
    interpolate ionospheric refraction and attenuation in the specified time range.

    :param dt_start: Start date/time of the model.
    :param dt_end: End date/time of the model.
    :param position: Geographical position of an observer. Must be a tuple containing
                     latitude [deg], longitude [deg], and elevation [m].
    :param mph: Number of static models per hour.
    :param nside: Resolution of healpix grid.
    :param dbot: Lower limit in [km] of the D layer of the ionosphere.
    :param dtop: Upper limit in [km] of the D layer of the ionosphere.
    :param ndlayers: Number of sub-layers in the D layer for intermediate calculations.
    :param fbot: Lower limit in [km] of the F layer of the ionosphere.
    :param ftop: Upper limit in [km] of the F layer of the ionosphere.
    :param nflayers: Number of sub-layers in the F layer for intermediate calculations.
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt_start: datetime,
            dt_end: datetime,
            position: Tuple[float, float, float],
            mph: int = 1,
            nside: int = 128,
            dbot: float = 60,
            dtop: float = 90,
            ndlayers: int = 10,
            fbot: float = 150,
            ftop: float = 500,
            nflayers: int = 30,
            _autocalc: bool = True,
    ):
        if not isinstance(dt_start, datetime) or not isinstance(dt_end, datetime):
            raise ValueError("Parameters dt_start and dt_end must be datetime objects.")
        self.dt_start = dt_start
        self.dt_end = dt_end
        nhours = (dt_end - dt_start).total_seconds() / 3600
        nmodels = int(nhours * mph)
        tdelta = timedelta(hours=nhours / nmodels)
        self._dts = np.asarray(
            [dt_start + tdelta * i for i in range(nmodels + 1)]
        ).astype(datetime)

        self.dbot = dbot
        self.dtop = dtop
        self.ndlayers = ndlayers
        self.fbot = fbot
        self.ftop = ftop
        self.nflayers = nflayers

        self.position = position
        self.mph = mph
        self.nside = nside
        self.models = []
        if _autocalc:
            for dt in tqdm(self._dts, desc="Calculating time frames"):
                self.models.append(
                    IonFrame(
                        dt,
                        position,
                        nside,
                        dbot,
                        dtop,
                        ndlayers,
                        fbot,
                        ftop,
                        nflayers,
                        _pbar=False,
                        _autocalc=_autocalc,
                    )
                )

    def save(self, directory: str = None, name: str = None):
        """
        Save the model to a file.

        :param directory: Path to directory to save the model.
        :param name: Name of the file (extension is not required).
        """
        import h5py

        filename = (
                f"ionmodel_{self.dt_start.year:04d}{self.dt_start.month:02d}"
                + f"{self.dt_start.day:02d}{self.dt_start.hour:02d}"
                + f"{self.dt_start.minute:02d}{self.dt_end.year:04d}"
                + f"{self.dt_end.month:02d}{self.dt_end.day:02d}"
                + f"{self.dt_end.hour:02d}{self.dt_end.minute:02d}"
        )
        directory = directory or "ion_models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        name = name or filename
        name = os.path.join(directory, name)
        if not name.endswith(".h5"):
            name += ".h5"

        file = h5py.File(name, mode="w")

        meta = file.create_dataset("meta", shape=(0,))
        meta.attrs["position"] = self.position
        meta.attrs["dt_start"] = self.dt_start.strftime("%Y-%m-%d %H:%M")
        meta.attrs["dt_end"] = self.dt_end.strftime("%Y-%m-%d %H:%M")
        meta.attrs["nside"] = self.nside
        meta.attrs["mph"] = self.mph
        meta.attrs["dbot"] = self.dbot
        meta.attrs["dtop"] = self.dtop
        meta.attrs["nlayers"] = self.ndlayers
        meta.attrs["hbot"] = self.fbot
        meta.attrs["htop"] = self.ftop
        meta.attrs["nlayers"] = self.nflayers

        for model in self.models:
            model.write_self_to_file(file)
        file.close()

    @classmethod
    def load(cls, path: str) -> "IonModel":
        """
        Load a model from file.

        :param path: Path to a file (file extension is not required).
        :return: :class:`IonModel` recovered from a file.
        """
        import h5py

        if not path.endswith(".h5"):
            path += ".h5"
        with h5py.File(path, mode="r") as file:
            groups = list(file.keys())
            try:
                groups.remove("meta")
            except ValueError:
                raise RuntimeError("The file is not an IonModel object.")

            if len(groups) <= 1:
                raise RuntimeError(
                    "File contains more less than two models. "
                    + "Consider reading it with IonFrame class."
                )
            meta = file.get("meta")
            obj = cls(
                _autocalc=False,
                dt_start=datetime.strptime(meta.attrs["dt_start"], "%Y-%m-%d %H:%M"),
                dt_end=datetime.strptime(meta.attrs["dt_end"], "%Y-%m-%d %H:%M"),
                position=meta.attrs["position"],
                nside=meta.attrs["nside"],
                mph=meta.attrs["mph"],
                dbot=meta.attrs["dbot"],
                dtop=meta.attrs["dtop"],
                ndlayers=meta.attrs["nlayers"],
                fbot=meta.attrs["hbot"],
                ftop=meta.attrs["htop"],
                nflayers=meta.attrs["nlayers"],
            )
            for group in groups:
                grp = file[group]
                obj.models.append(IonFrame.read_self_from_file(grp))
            return obj

    def _lr_ind(self, dt: datetime) -> [int, int]:
        """
        Calculates indices on the left and on the right of the specified date
        """
        if (dt - self.dt_start).total_seconds() < 0 or (
                self.dt_end - dt
        ).total_seconds() < 0:
            raise ValueError(
                f"Datetime must be within precalculated range "
                + "{str(self.dt_start)} - {str(self.dt_end)}."
            )
        idx = np.searchsorted(self._dts, dt)
        if idx == 0:
            return [idx, idx]
        return [idx - 1, idx]

    def _parallel_calc(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime] | np.ndarray,
            funcs: List[Callable],
            pbar_desc: str,
            *args,
            **kwargs,
    ) -> float | np.ndarray:
        """
        Sends methods either to serial or parallel calculation routines based on type of dt.
        """
        if (isinstance(dt, list) or isinstance(dt, np.ndarray)) and len(dt) > 1:
            idx = [self._lr_ind(i) for i in dt]
            dts = [self._dts[i] for i in idx]
            dts = [np.append(dts[i], dt[i]) for i in range(len(dts))]
            funcs = [[funcs[i[0]], funcs[i[1]]] for i in idx]
            return calc_interp_val_par(el, az, funcs, dts, pbar_desc, *args, **kwargs)
        else:
            idx = self._lr_ind(dt)
            dt1, dt2 = self._dts[idx]
            funcs = [funcs[idx[0]], funcs[idx[1]]]
            return calc_interp_val(el, az, funcs, [dt1, dt2, dt], *args, **kwargs)

    def ded(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            layer: int | None = None,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores.
        :param layer: A number of sublayer to calculate. If None - an average value over all layers will be calculated.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Electron density in [m^-3] in the D layer of the ionosphere.
        """
        funcs = [m.dlayer.ed for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def det(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            layer: int | None = None,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores.
        :param layer: A number of sublayer to calculate. If None - an average value over all layers will be calculated.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Electron temperature in [K] in the D layer of the ionosphere.
        """
        funcs = [m.dlayer.et for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def fed(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            layer: int | None = None,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores.
        :param layer: A number of sublayer to calculate. If None - an average value over all layers will be calculated.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Electron density in [m^-3] in the F layer of the ionosphere.
        """
        funcs = [m.flayer.ed for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def fet(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            layer: int = None,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores.
        :param layer: A number of sublayer to calculate. If None - an average value over all layers will be calculated.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Electron temperature in [K] in the F layer of the ionosphere.
        """
        funcs = [m.flayer.et for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def at(self, dt: datetime, recalc: bool = False) -> IonFrame:
        """
        :param dt: Date/time of the frame.
        :param recalc: If True - the :class:`IonFrame` object will be precisely calculated. If False - an interpolation
                       of two closest frames will be used.
        :return: :class:`IonFrame` at specified time.
        """
        if dt in self._dts:
            idx = np.argwhere(self._dts == dt)
            return self.models[idx[0][0]]
        obj = IonFrame(
            dt=dt,
            position=self.position,
            nside=self.nside,
            dbot=self.dbot,
            dtop=self.dtop,
            ndlayers=self.ndlayers,
            fbot=self.fbot,
            ftop=self.ftop,
            nflayers=self.nflayers,
            _pbar=False,
            _autocalc=recalc,
        )
        if recalc:
            return obj
        else:
            idx = self._lr_ind(dt)
            obj.dlayer.edens = interp_val(
                self.models[idx[0]].dlayer.edens,
                self.models[idx[1]].dlayer.edens,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            obj.dlayer.etemp = interp_val(
                self.models[idx[0]].dlayer.etemp,
                self.models[idx[1]].dlayer.etemp,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            obj.flayer.f_e_density = interp_val(
                self.models[idx[0]].flayer.f_e_density,
                self.models[idx[1]].flayer.f_e_density,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            obj.flayer.f_e_temp = interp_val(
                self.models[idx[0]].flayer.f_e_temp,
                self.models[idx[1]].flayer.f_e_temp,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            return obj

    def atten(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            freq: float | np.ndarray,
            col_freq: str = "default",
            troposphere: bool = True,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores. Requires `freq` to be a single float.
        :param freq: Frequency of observation(s) in [MHz]. If  - the calculation will be performed in parallel on all
                     available cores. Requires `dt` to be a single datetime object.
        :param col_freq: Collision frequency model. Available options: 'default', 'nicolet', 'setty', 'aggrawal',
                         or float in Hz.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Attenuation factor at given sky coordinates, time and frequency of observation. Output is the
                 attenuation factor between 0 (total attenuation) and 1 (no attenuation).
        """
        if is_iterable(freq) and not is_iterable(dt):
            model = self.at(dt)
            return model.atten(el, az, freq, _pbar_desc=_pbar_desc, col_freq=col_freq, troposphere=troposphere)
        elif is_iterable(dt) and not is_iterable(freq):
            funcs = [m.dlayer.atten for m in self.models]
            return self._parallel_calc(
                el,
                az,
                dt,
                funcs,
                _pbar_desc,
                freq,
                col_freq=col_freq,
                troposphere=troposphere,
            )
        else:
            raise ValueError("Both datetime and frequency cannot be iterables at the same call.")

    def refr(
            self,
            el: float | np.ndarray,
            az: float | np.ndarray,
            dt: datetime | List[datetime],
            freq: float | np.ndarray,
            troposphere: bool = True,
            _pbar_desc: str | None = None,
    ) -> float | np.ndarray:
        """
        :param el: Elevation of observation(s) in [deg].
        :param az: Azimuth of observation(s) in [deg].
        :param dt: Datetime of observation(s). If list - the calculation will be performed in parallel on all available
                   cores. Requires `freq` to be a single float.
        :param freq: Frequency of observation(s) in [MHz]. If  - the calculation will be performed in parallel on all
                     available cores. Requires `dt` to be a single datetime object.
        :param troposphere: If True - the troposphere refraction correction will be applied before calculation.
        :param _pbar_desc: Description of progress bar. If None - the progress bar will not appear.
        :return: Refraction angle in [deg] at given sky coordinates, time and frequency of observation.
        """
        funcs = [m.flayer.refr for m in self.models]
        return self._parallel_calc(
            el, az, dt, funcs, _pbar_desc, freq, troposphere=troposphere
        )

    @staticmethod
    def troprefr(el: float | np.ndarray) -> float | np.ndarray:
        """
        Approximation of the refraction in the troposphere recommended by the ITU-R:
        https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.834-7-201510-S!!PDF-E.pdf

        :param el: Elevation of observation(s) in [deg].
        :return: Refraction in the troposphere in [deg].
        """
        return IonFrame.troprefr(el)

    def _nframes2dts(self, nframes: int | None) -> ndarray:
        """
        Returns a list of datetimes for animation based on specified number of frames (fps * duration).
        """
        if nframes is None:
            dts = self._dts
        else:
            tdelta = timedelta(
                seconds=(self.dt_end - self.dt_start).total_seconds() / nframes
            )
            dts = np.asarray(
                [self.dt_start + tdelta * i for i in range(nframes + 1)]
            ).astype(datetime)
        return dts

    def _time_animation(
            self,
            func: Callable,
            name: str,
            freq: float | None = None,
            gridsize: int = 100,
            fps: int = 20,
            duration: int = 5,
            savedir: str = "animations/",
            title: str | None = None,
            barlabel: str | None = None,
            plotlabel: str | None = None,
            dpi: int = 300,
            cmap: str = "viridis",
            pbar_label: str = "",
            nancolor: str = "black",
            local_time: int | None = None,
    ):
        """
        Abstract method for generating animations.
        """
        print(
            TextColor.BOLD
            + TextColor.BLUE
            + "Animation making procedure started"
            + f" [{pbar_label}]"
            + TextColor.END
            + TextColor.END
        )
        el, az = elaz_mesh(gridsize)
        nframes = duration * fps
        dts = self._nframes2dts(nframes)
        extra_args = [arg for arg in [freq] if arg is not None]
        data = np.array(
            func(el, az, dts, *extra_args, _pbar_desc="[1/3] Calculating data")
        )

        cbmax = np.nanmax(data)
        cbmin = np.nanmin(data)

        tmpdir = tempfile.mkdtemp()
        nproc = np.min([cpu_count(), len(dts)])
        plot_data = [(np.deg2rad(az), 90 - el, data[i]) for i in range(len(data))]
        plot_saveto = [os.path.join(tmpdir, str(i).zfill(6)) for i in range(len(data))]
        with Pool(processes=nproc) as pool:
            list(
                tqdm(
                    pool.imap(
                        polar_plot_star,
                        zip(
                            plot_data,
                            dts,
                            itertools.repeat(self.position),
                            itertools.repeat(freq),
                            itertools.repeat(title),
                            itertools.repeat(barlabel),
                            itertools.repeat(plotlabel),
                            itertools.repeat((cbmin, cbmax)),
                            plot_saveto,
                            itertools.repeat(dpi),
                            itertools.repeat(cmap),
                            itertools.repeat(None),
                            itertools.repeat(nancolor),
                            itertools.repeat(local_time),
                        ),
                    ),
                    desc="[2/3] Rendering frames",
                    total=len(dts),
                )
            )
        desc = "[3/3] Rendering video"
        pic2vid(tmpdir, name, fps=fps, desc=desc, savedir=savedir)

        shutil.rmtree(tmpdir)
        return

    def animate_atten_vs_time(self, name: str, freq: float, **kwargs):
        """
        Generates an animation of attenuation factor change with time.

        :param name: Name of file.
        :param freq: Frequency of observation.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        self._time_animation(
            self.atten,
            name,
            freq=freq,
            pbar_label="D layer attenuation",
            **kwargs,
        )

    def animate_refr_vs_time(self, name: str, freq: float, cmap: str = "viridis_r", **kwargs):
        """
        Generates an animation of refraction angle change with time.

        :param name: Name of file.
        :param freq: Frequency of observation.
        :param cmap: Matplotlib colormap to use in plot.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$deg$"
        self._time_animation(
            self.refr,
            name,
            freq=freq,
            barlabel=barlabel,
            pbar_label="F layer refraction",
            cmap=cmap,
            **kwargs,
        )

    def animate_ded_vs_time(self, name: str, **kwargs):
        """
        Generates an animation of change of electron temperature in the D layer with time.

        :param name: Name of file.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$m^{-3}$"
        self._time_animation(
            self.ded,
            name,
            barlabel=barlabel,
            pbar_label="D layer electron density",
            **kwargs,
        )

    def animate_det_vs_time(self, name: str, **kwargs):
        """
        Generates an animation of change of electron density in the D layer with time.

        :param name: Name of file.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$^\circ K$"
        self._time_animation(
            self.det,
            name,
            barlabel=barlabel,
            pbar_label="D layer electron temperature",
            **kwargs,
        )

    def animate_fed_vs_time(self, name: str, **kwargs):
        """
        Generates an animation of change of electron density in the F layer with time.

        :param name: Name of file.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$m^{-3}$"
        self._time_animation(
            self.fed,
            name,
            barlabel=barlabel,
            pbar_label="F layer electron density",
            **kwargs,
        )

    def animate_fet_vs_time(self, name: str, **kwargs):
        """
        Generates an animation of change of electron temperature in the F layer with time.

        :param name: Name of file.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$^\circ K$"
        self._time_animation(
            self.fet,
            name,
            barlabel=barlabel,
            pbar_label="F layer electron temperature",
            **kwargs,
        )
