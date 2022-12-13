from __future__ import annotations

import itertools
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Callable
import warnings

import numpy as np
from iricore import readapf107
from numpy import ndarray
from tqdm import tqdm

from .IonFrame import IonFrame
from .modules.helpers import elaz_mesh, TextColor, pic2vid, get_atten_from_frame, get_refr_from_frame
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
    :param mpf: Number of minutes per frame.
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
    :param _autocalc: If True - the model will be calculated immediately after definition.
    """

    def __init__(
            self,
            dt_start: datetime,
            dt_end: datetime,
            position: List[float, float, float],
            mpf: int = 15,
            nside: int = 64,
            dbot: float = 60,
            dtop: float = 90,
            ndlayers: int = 100,
            fbot: float = 150,
            ftop: float = 500,
            nflayers: int = 100,
            iriversion: int = 20,
            echaim: bool = False,
            _autocalc: bool = True,
    ):
        if not isinstance(dt_start, datetime) or not isinstance(dt_end, datetime):
            raise ValueError("Parameters dt_start and dt_end must be datetime objects.")
        if position[2] != 0:
            position[2] = 0
            warnings.warn("The current model does not support non zero altitude in instrument position. Setting "
                          "instrument altitude to zero.", RuntimeWarning, stacklevel=2)

        self.dt_start = dt_start
        self.dt_end = dt_end
        nhours = (dt_end - dt_start).total_seconds() / 3600
        nmodels = int(nhours * 60 / mpf)
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
        self.mpf = mpf
        self.nside = nside
        self.iriversion = iriversion
        self.models = []

        if _autocalc:
            nproc = np.min([cpu_count(), nmodels])
            # nproc = 1
            pool = Pool(processes=nproc)
            apf107_args = readapf107(self.iriversion)

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
                        iriversion,
                        echaim,
                        _pbar=False,
                        _autocalc=_autocalc,
                        _pool=pool,
                        _apf107_args=apf107_args,
                    )
                )
            pool.close()

    def save(self, saveto: str = "./ionmodel"):
        """
        Save the model to a file.

        :param saveto: Path to directory with name to save the model.
        """
        import h5py

        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)

        if not saveto.endswith(".h5"):
            saveto += ".h5"

        file = h5py.File(saveto, mode="w")

        meta = file.create_dataset("meta", shape=(0,))
        meta.attrs["position"] = self.position
        meta.attrs["dt_start"] = self.dt_start.strftime("%Y-%m-%d %H:%M")
        meta.attrs["dt_end"] = self.dt_end.strftime("%Y-%m-%d %H:%M")
        meta.attrs["nside"] = self.nside
        meta.attrs["mpf"] = self.mpf
        meta.attrs["dbot"] = self.dbot
        meta.attrs["dtop"] = self.dtop
        meta.attrs["nlayers"] = self.ndlayers
        meta.attrs["hbot"] = self.fbot
        meta.attrs["htop"] = self.ftop
        meta.attrs["nlayers"] = self.nflayers
        meta.attrs["iriversion"] = self.iriversion

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
                mpf=meta.attrs["mpf"],
                dbot=meta.attrs["dbot"],
                dtop=meta.attrs["dtop"],
                ndlayers=meta.attrs["nlayers"],
                fbot=meta.attrs["hbot"],
                ftop=meta.attrs["htop"],
                nflayers=meta.attrs["nlayers"],
                iriversion=meta.attrs["iriversion"],
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
            obj.flayer.edens = interp_val(
                self.models[idx[0]].flayer.edens,
                self.models[idx[1]].flayer.edens,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            obj.flayer.etemp = interp_val(
                self.models[idx[0]].flayer.etemp,
                self.models[idx[1]].flayer.etemp,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            return obj

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
            saveto: str,
            freq: float | None = None,
            gridsize: int = 100,
            fps: int = 20,
            duration: int = 5,
            title: str | None = None,
            barlabel: str | None = None,
            plotlabel: str | None = None,
            dpi: int = 300,
            cmap: str = "viridis",
            pbar_label: str = "",
            nancolor: str = "black",
            infcolor: str = "white",
            local_time: int | None = None,
            codec: str = "libx264",
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
        frames = [self.at(dt_) for dt_ in dts]
        nproc = np.min([cpu_count(), len(dts)])
        pool = Pool(processes=nproc)
        data = np.array(list(
            tqdm(
                pool.imap(
                    func,
                    zip(
                        frames,
                        itertools.repeat(el),
                        itertools.repeat(az),
                        itertools.repeat(freq),
                    ),
                ),
                desc="[1/3] Calculating data",
                total=len(dts),
            )
        ))

        cbmin, cbmax = np.nanmin(data[data != -np.inf]), np.nanmax(data[data != np.inf])
        tmpdir = tempfile.mkdtemp()
        plot_data = [(np.deg2rad(az), 90 - el, data[i]) for i in range(len(data))]
        plot_saveto = [os.path.join(tmpdir, str(i).zfill(6)) for i in range(len(data))]
        del data

        try:
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
                            itertools.repeat(infcolor),
                            itertools.repeat(local_time),
                        ),
                    ),
                    desc="[2/3] Rendering frames",
                    total=len(dts),
                )
            )
            del plot_data
            pool.close()
            desc = "[3/3] Rendering video"
            pic2vid(tmpdir, saveto, fps=fps, desc=desc, codec=codec)

        except Exception as e:
            pool.close()
            print(e)

        shutil.rmtree(tmpdir)
        return

    def animate_atten_vs_time(self, freq: float, saveto: str = "./atten_vs_time", **kwargs):
        """
        Generates an animation of attenuation factor change with time.

        :param saveto: Path to save a file including name.
        :param freq: Frequency of observation.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        self._time_animation(
            get_atten_from_frame,
            saveto,
            freq=freq,
            pbar_label="D layer attenuation",
            **kwargs,
        )

    def animate_refr_vs_time(self, freq: float, saveto: str = "./refr_vs_time", cmap: str = "viridis_r", **kwargs):
        """
        Generates an animation of refraction angle change with time.

        :param saveto: Path to save a file including name.
        :param freq: Frequency of observation.
        :param cmap: Matplotlib colormap to use in plot.
        :param kwargs: See `dionpy.plot_kwargs`.
        """
        barlabel = r"$deg$"
        self._time_animation(
            get_refr_from_frame,
            saveto,
            freq=freq,
            barlabel=barlabel,
            pbar_label="F layer refraction",
            cmap=cmap,
            **kwargs,
        )
