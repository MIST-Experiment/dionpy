from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Sequence, Literal

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from .IonFrame import IonFrame
from .modules.helpers import altaz_mesh, pic2vid, open_save_file
from .modules.parallel import interp_val
from .modules.plotting import polar_plot


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
            dt_start: datetime,
            dt_end: datetime,
            position: Sequence[float, float, float],
            mpf: int = 15,
            nside: int = 64,
            hbot: float = 60,
            htop: float = 500,
            nlayers: int = 500,
            rdeg_offset: float = 5,
            iriversion: Literal[16, 20] = 20,
            echaim: bool = False,
            autocalc: bool = True,
    ):
        if not isinstance(dt_start, datetime) or not isinstance(dt_end, datetime):
            raise ValueError("Parameters dt_start and dt_end must be datetime objects.")

        self.dt_start = dt_start
        self.dt_end = dt_end
        nhours = (dt_end - dt_start).total_seconds() / 3600
        nmodels = int(nhours * 60 / mpf)
        tdelta = timedelta(hours=nhours / nmodels)
        self._dts = np.asarray(
            [dt_start + tdelta * i for i in range(nmodels + 1)]
        ).astype(datetime)

        self.hbot = hbot
        self.htop = htop
        self.nlayers = nlayers
        self.rdeg_offset = rdeg_offset
        self.echaim = echaim

        self.position = position
        self.mpf = mpf
        self.nside = nside
        self.iriversion = iriversion
        self.frames = []

        if autocalc:
            nproc = np.min([cpu_count(), nmodels])
            pool = Pool(processes=nproc)

            for dt in tqdm(self._dts, desc="Calculating time frames"):
                self.frames.append(
                    IonFrame(
                        dt=dt,
                        position=position,
                        nside=nside,
                        hbot=hbot,
                        htop=htop,
                        nlayers=nlayers,
                        rdeg_offset=rdeg_offset,
                        iriversion=iriversion,
                        echaim=echaim,
                        autocalc=autocalc,
                        _pool=pool,
                    )
                )
            pool.close()

    def __str__(self):
        frame_str = str(self.frames[0])
        frame_str = "\n".join(frame_str.split("\n")[2:])

        return (
                f"IonModel instance\n"
                f"Start date:\t{self.dt_start.strftime('%d %b %Y %H:%M:%S')} UTC\n"
                f"End date:\t{self.dt_end.strftime('%d %b %Y %H:%M:%S')} UTC\n"
                f"Minutes per frame:\t{self.mpf}\n"
                "" + frame_str
        )

    def at(self, dt: datetime, recalc: bool = False) -> IonFrame:
        """
        :param dt: Date/time of the frame.
        :param recalc: If True - the :class:`IonFrame` object will be precisely calculated. If False - an interpolation
                       of two closest frames will be used.
        :return: :class:`IonFrame` at specified time.
        """
        if dt in self._dts:
            idx = np.argwhere(self._dts == dt)
            return self.frames[idx[0][0]]
        frame_dict = self.frames[0].get_init_dict()
        del frame_dict['dt']
        obj = IonFrame(
            dt=dt,
            **frame_dict,
            autocalc=recalc,
        )
        if recalc:
            return obj
        else:
            idx = self._lr_ind(dt)
            obj.layer.edens = interp_val(
                self.frames[idx[0]].layer.edens,
                self.frames[idx[1]].layer.edens,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            obj.layer.etemp = interp_val(
                self.frames[idx[0]].layer.etemp,
                self.frames[idx[1]].layer.etemp,
                self._dts[idx[0]],
                self._dts[idx[1]],
                dt,
            )
            return obj

    def save(self, saveto: str = "./ionmodel"):
        """
        Save the model to a file.

        :param saveto: Path to directory with name to save the model.
        """
        with open_save_file(saveto) as file:
            meta = file.create_dataset("meta", shape=(0,))
            meta.attrs["position"] = self.position
            meta.attrs["dt_start"] = self.dt_start.strftime("%Y-%m-%d %H:%M")
            meta.attrs["dt_end"] = self.dt_end.strftime("%Y-%m-%d %H:%M")
            meta.attrs["nside"] = self.nside
            meta.attrs["mpf"] = self.mpf
            meta.attrs["hbot"] = self.hbot
            meta.attrs["htop"] = self.htop
            meta.attrs["nlayers"] = self.nlayers
            meta.attrs["rdeg_offset"] = self.rdeg_offset
            meta.attrs["iriversion"] = self.iriversion
            meta.attrs["echaim"] = self.echaim

            for model in self.frames:
                model.write_self_to_file(file)

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
                    "File contains more less than two frames. "
                    + "Consider reading it with IonFrame class."
                )
            meta = file.get("meta")
            meta_attrs = dict(meta.attrs)
            del meta_attrs['dt_start']
            del meta_attrs['dt_end']
            obj = cls(
                autocalc=False,
                dt_start=datetime.strptime(meta.attrs["dt_start"], "%Y-%m-%d %H:%M"),
                dt_end=datetime.strptime(meta.attrs["dt_end"], "%Y-%m-%d %H:%M"),
                **meta_attrs
            )
            for group in groups:
                grp = file[group]
                obj.frames.append(IonFrame.read_self_from_file(grp))
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
        # noinspection PyTypeChecker
        idx = np.searchsorted(self._dts, dt)
        if idx == 0:
            return [idx, idx]
        return [idx - 1, idx]

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

    @staticmethod
    def _render_polar_plot_frames(alt: np.ndarray, az: np.ndarray, data: np.ndarray, dts: np.ndarray,
                                  plot_kwargs: dict, desc: str):
        cbmin, cbmax = np.nanmin(data[data != -np.inf]), np.nanmax(data[data != np.inf])
        tmpdir = tempfile.mkdtemp()
        plot_kwargs['cblim'] = (cbmin, cbmax)
        try:
            for i in tqdm(range(data.shape[0]), desc=desc):
                plot_kwargs['saveto'] = os.path.join(tmpdir, str(i).zfill(6))
                polar_plot((np.deg2rad(az), 90 - alt, data[i, ...]), dts[i], **plot_kwargs)
        except Exception as e:
            shutil.rmtree(tmpdir)
            raise e
        return tmpdir

    def animate(
            self,
            target: str | List["str"],
            saveto: str = "./",
            freq: float | None = None,
            gridsize: int = 100,
            fps: int = 20,
            duration: int = 5,
            codec: str = "libx264",
            **plot_kwargs,
    ):
        """
        Calculates the dynamic evolution of specified parameters and generates an animated visualization.

        :param target: Parameter or list of parameters to calculate. Options include:
                       "atten" - Attenuation (integrated)
                       "refr" - Refraction (integrated)
                       "emiss" - Emission (integrated)
                       "edens" - Electron density (height average)
                       "etemp" - Electron temperature (height average)
        :param saveto: Location of the output files. Defaults to the script execution directory ("./").
        :param freq: Frequency of observation.
        :param gridsize: Grid resolution of calculated data.
        :param fps: Frames per second.
        :param duration: Total duration of the video in seconds.
        :param codec: Specify the codec for ffmpeg to use.
        """
        print("Animation making procedure started")
        target = [target] if isinstance(target, str) else target
        alt, az = altaz_mesh(gridsize)
        dts = self._nframes2dts(duration * fps)
        frames = [self.at(dt_) for dt_ in dts]
        nframes = len(frames)
        data_dict = {
            'atten': np.empty((nframes, *alt.shape)),
            'refr': np.empty((nframes, *alt.shape)),
            'emiss': np.empty((nframes, *alt.shape)),
            'edens': np.empty((nframes, *alt.shape)),
            'etemp': np.empty((nframes, *alt.shape))
        }
        plot_data_dict = {
            'atten': dict(cmap="plasma", barlabel=None),
            'refr': dict(cmap="plasma_r", barlabel=r"deg"),
            'emiss': dict(cmap="plasma", barlabel=r"K"),
            'edens': dict(cmap="plasma", barlabel=r"$m^{-3}$"),
            'etemp': dict(cmap="plasma", barlabel=r"$m^{-3}$"),
        }
        nproc = np.min([cpu_count(), len(dts)])
        pool = Pool(nproc)

        try:
            print("Calculating data")
            if "atten" in target or "refr" in target or "emiss" in target:
                if freq is None:
                    raise ValueError("Please specify the frequency for the simulation.")
                for i, frame in enumerate(tqdm(frames, desc="Raytracing frames")):
                    data_dict['refr'][i, ...], data_dict['atten'][i, ...], data_dict['emiss'][i, ...] = frame(alt, az,
                                                                                                              freq,
                                                                                                              _pool=pool)

            if "edens" in target:
                for i, frame in enumerate(tqdm(frames, desc="Interpolating ed")):
                    data_dict['edens'][i, ...] = frame.layer.ed(alt, az)
            if "etemp" in target:
                for i, frame in enumerate(tqdm(frames, desc="Interpolating et")):
                    data_dict['etemp'][i, ...] = frame.layer.et(alt, az)

            plot_kwargs['pos'] = self.position
            plot_kwargs['freq'] = freq

            for key in data_dict:
                if key in target:
                    if 'cmap' not in plot_kwargs.keys():
                        plot_kwargs['cmap'] = plot_data_dict[key]['cmap']
                    if 'barlabel' not in plot_kwargs.keys():
                        plot_kwargs['barlabel'] = plot_data_dict[key]['barlabel']
                    tmpdir = self._render_polar_plot_frames(alt, az, data_dict[key], dts, plot_kwargs,
                                                            desc=f"Rendering {key} frames")
                    pic2vid(tmpdir, saveto + key, fps=fps, desc=f"Rendering {key} animation", codec=codec)
                    shutil.rmtree(tmpdir)
        except Exception as e:
            pool.close()
            raise e
        return
