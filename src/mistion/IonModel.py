import glob
import os
from typing import Tuple
import tempfile
import shutil
from ffmpeg_progress_yield import FfmpegProgress

import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from .SingleTimeModel import SingleTimeModel
from datetime import datetime, timedelta

from .modules.helpers import elaz_mesh, TextColor, polar_plot, calc_interp_val


class IonModel:
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
        autocalc: bool = True,
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

        self.position = position
        self.mph = mph
        self.nside = nside
        self.models = []
        if autocalc:
            for dt in tqdm(self._dts, desc="Calculating time frames"):
                self.models.append(
                    SingleTimeModel(
                        dt,
                        position,
                        nside,
                        dbot,
                        dtop,
                        ndlayers,
                        fbot,
                        ftop,
                        nflayers,
                        autocalc,
                        pbar=False,
                    )
                )

    def save(self, directory=None, name=None):
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

        for model in self.models:
            model.write_self_to_file(file)
        file.close()

    @classmethod
    def load(cls, path: str):
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
                    + "Consider reading it with SingleTimeModel class."
                )
            meta = file.get("meta")
            obj = cls(
                autocalc=False,
                dt_start=datetime.strptime(meta.attrs["dt_start"], "%Y-%m-%d %H:%M"),
                dt_end=datetime.strptime(meta.attrs["dt_end"], "%Y-%m-%d %H:%M"),
                position=meta.attrs["position"],
                nside=meta.attrs["nside"],
                mph=meta.attrs["mph"],
            )
            for group in groups:
                grp = file[group]
                obj.models.append(SingleTimeModel.read_self_from_file(grp))
            return obj

    def _lr_ind(self, dt):
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

    def ded(self, el, az, dt, layer=None):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].dlayer.ded(el, az, layer)
        data2 = self.models[idx[1]].dlayer.ded(el, az, layer)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    def det(self, el, az, dt, layer=None):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].dlayer.det(el, az, layer)
        data2 = self.models[idx[1]].dlayer.det(el, az, layer)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    def fed(self, el, az, dt, freq, col_freq="default", layer=None):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].flayer.fed(el, az, freq, col_freq, layer)
        data2 = self.models[idx[1]].flayer.fed(el, az, freq, col_freq, layer)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    def fet(self, el, az, dt, freq, col_freq="default", layer=None):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].flayer.fet(el, az, freq, col_freq, layer)
        data2 = self.models[idx[1]].flayer.fet(el, az, freq, col_freq, layer)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    def datten(self, el, az, dt, freq, col_freq="default", troposphere=True):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].datten(el, az, freq, col_freq, troposphere)
        data2 = self.models[idx[1]].datten(el, az, freq, col_freq, troposphere)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    def frefr(self, el, az, dt, freq, troposphere=True):
        idx = self._lr_ind(dt)
        dt1, dt2 = self._dts[idx]
        data1 = self.models[idx[0]].frefr(el, az, freq, troposphere)
        data2 = self.models[idx[1]].frefr(el, az, freq, troposphere)
        return calc_interp_val(data1, data2, dt1, dt2, dt)

    @staticmethod
    def troprefr(el=None):
        return SingleTimeModel.troprefr(el)

    @staticmethod
    def _pic2vid(
        imdir: str, vidname: str, savedir: str = "animations", fps: int = 20, desc=None
    ):
        if not vidname.endswith(".mp4"):
            vidname += ".mp4"
        desc = desc or "Rendering video"
        cmd = [
            "ffmpeg",
            "-r",
            f"{fps}",
            "-i",
            os.path.join(imdir, "%06d.png"),
            "-vcodec",
            "libx265",
            "-y",
            os.path.join(savedir, vidname),
        ]
        print(" ".join(cmd))
        # return
        ff = FfmpegProgress(cmd)
        with tqdm(total=100, position=1, desc=desc) as pbar:
            for progress in ff.run_command_with_progress():
                pbar.update(progress - pbar.n)

    def _nframes2dts(self, nframes):
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
        func,
        extra_args,
        name,
        gridsize=100,
        nframes=None,
        fps=20,
        title=None,
        barlabel=None,
        plotlabel=None,
        dpi=300,
        cmap="viridis",
        label="",
    ):
        print(
            TextColor.BOLD
            + TextColor.BLUE
            + "Animation making procedure started"
            + f" [{label}]"
            + TextColor.END
            + TextColor.END
        )
        el, az = elaz_mesh(gridsize)
        nframes = nframes or len(self._dts)
        dts = self._nframes2dts(nframes)
        it = tqdm(dts, desc="[1/3] Calculating data")
        data = np.array([func(el, az, dt, *extra_args) for dt in it])
        cbmax = np.max(data)
        cbmin = np.min(data)

        tmpdir = tempfile.mkdtemp()
        for i in tqdm(range(len(dts)), desc="[2/3] Rendering frames"):
            polar_plot(
                dts[i],
                (np.deg2rad(az), 90 - el, data[i]),
                title=title,
                barlabel=barlabel,
                plotlabel=plotlabel,
                cblim=(cbmin, cbmax),
                saveto=os.path.join(tmpdir, str(i).zfill(6)),
                dpi=dpi,
                cmap=cmap,
            )
        desc = "[3/3] Rendering video"
        self._pic2vid(tmpdir, name, fps=fps, desc=desc)

        shutil.rmtree(tmpdir)
        return

    def animate_datten_vs_time(
        self,
        name,
        freq,
        gridsize=100,
        fps=20,
        nframes=None,
        title=None,
        barlabel=None,
        plotlabel=None,
    ):
        title = title or f"Attenuation factor at {freq / 1e6:.2f} MHz"
        self._time_animation(
            self.datten,
            [freq],
            name,
            gridsize,
            fps=fps,
            nframes=nframes,
            title=title,
            barlabel=barlabel,
            plotlabel=plotlabel,
            dpi=300,
            cmap="viridis",
            label="D layer attenuation",
        )

    def animate_frefr_vs_time(
        self,
        name,
        freq,
        gridsize=100,
        fps=20,
        nframes=None,
        title=None,
        barlabel=None,
        plotlabel=None,
    ):
        title = title or f"Refraction angle at {freq / 1e6:.2f} MHz"
        self._time_animation(
            self.frefr,
            [freq],
            name,
            gridsize,
            fps=fps,
            nframes=nframes,
            title=title,
            barlabel=barlabel,
            plotlabel=plotlabel,
            dpi=300,
            cmap="viridis_r",
            label="F layer refraction",
        )
