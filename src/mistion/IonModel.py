import itertools
import os
from multiprocessing import Pool, cpu_count
from typing import Tuple
import tempfile
import shutil
from ffmpeg_progress_yield import FfmpegProgress

import numpy as np
from tqdm import tqdm

from .IonFrame import IonFrame
from datetime import datetime, timedelta

from .modules.helpers import (
    elaz_mesh,
    TextColor,
    calc_interp_val,
    calc_interp_val_par,
    polar_plot_star,
)


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
                        pbar=False,
                        _autocalc=_autocalc,
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
                    + "Consider reading it with IonFrame class."
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
                obj.models.append(IonFrame.read_self_from_file(grp))
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

    def _parallel_calc(self, el, az, dt, funcs, pbar_desc, *args, **kwargs):
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

    def ded(self, el, az, dt, layer=None, _pbar_desc=None):
        funcs = [m.dlayer.ded for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def det(self, el, az, dt, layer=None, _pbar_desc=None):
        funcs = [m.dlayer.det for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def fed(self, el, az, dt, layer=None, _pbar_desc=None):
        funcs = [m.flayer.fed for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def fet(self, el, az, dt, layer=None, _pbar_desc=None):
        funcs = [m.flayer.fet for m in self.models]
        return self._parallel_calc(el, az, dt, funcs, _pbar_desc, layer=layer)

    def datten(
        self, el, az, dt, freq, col_freq="default", troposphere=True, _pbar_desc=None
    ):
        funcs = [m.dlayer.datten for m in self.models]
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

    def frefr(self, el, az, dt, freq, troposphere=True, _pbar_desc=None):
        funcs = [m.flayer.frefr for m in self.models]
        return self._parallel_calc(
            el, az, dt, funcs, _pbar_desc, freq, troposphere=troposphere
        )

    @staticmethod
    def troprefr(el=None):
        return IonFrame.troprefr(el)

    @staticmethod
    def _pic2vid(
        imdir: str,
        vidname: str,
        savedir: str = "animations",
        fps: int = 20,
        desc=None,
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
        name,
        extra_args,
        gridsize=100,
        fps=20,
        duration=5,
        savedir="animations/",
        title=None,
        barlabel=None,
        plotlabel=None,
        dpi=300,
        cmap="viridis",
        pbar_label="",
    ):
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
        data = np.array(
            func(el, az, dts, *extra_args, _pbar_desc="[1/3] Calculating data")
        )

        cbmax = np.max(data)
        cbmin = np.min(data)

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
                            itertools.repeat(title),
                            itertools.repeat(barlabel),
                            itertools.repeat(plotlabel),
                            itertools.repeat((cbmin, cbmax)),
                            plot_saveto,
                            itertools.repeat(dpi),
                            itertools.repeat(cmap),
                        ),
                    ),
                    desc="[2/3] Rendering frames",
                    total=len(dts),
                )
            )
        desc = "[3/3] Rendering video"
        self._pic2vid(tmpdir, name, fps=fps, desc=desc, savedir=savedir)

        shutil.rmtree(tmpdir)
        return

    def animate_datten_vs_time(self, name, freq, title=None, barlabel=None, **kwargs):
        title = (
            title or r"Attenuation factor $(1 - f_{a})$ " + f"at {freq / 1e6:.2f} MHz"
        )
        self._time_animation(
            lambda *args, **kwargs: 1 - self.datten(*args, **kwargs),
            name,
            [freq],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis_r",
            pbar_label="D layer attenuation",
            **kwargs,
        )

    def animate_frefr_vs_time(self, name, freq, title=None, barlabel=None, **kwargs):
        title = title or r"Refraction $\delta \theta$ " + f"at {freq / 1e6:.2f} MHz"
        barlabel = barlabel or r"$deg$"
        self._time_animation(
            self.frefr,
            name,
            [freq],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis_r",
            pbar_label="F layer refraction",
            **kwargs,
        )

    def animate_ded_vs_time(self, name, title=None, barlabel=None, **kwargs):
        title = title or r"Electron density $n_e$ in the D layer"
        barlabel = barlabel or r"$m^{-3}$"
        self._time_animation(
            self.ded,
            name,
            [],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis",
            pbar_label="D layer electron density",
            **kwargs,
        )

    def animate_det_vs_time(self, name, title=None, barlabel=None, **kwargs):
        title = title or r"Electron temperature $T_e$ in the D layer"
        barlabel = barlabel or r"$^\circ K$"
        self._time_animation(
            self.det,
            name,
            [],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis",
            pbar_label="D layer electron temperature",
            **kwargs,
        )

    def animate_fed_vs_time(self, name, title=None, barlabel=None, **kwargs):
        title = title or r"Electron density $n_e$ in the F layer"
        barlabel = barlabel or r"$m^{-3}$"
        self._time_animation(
            self.fed,
            name,
            [],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis",
            pbar_label="F layer electron density",
            **kwargs,
        )

    def animate_fet_vs_time(self, name, title=None, barlabel=None, **kwargs):
        title = title or r"Electron temperature $T_e$ in the F layer"
        barlabel = barlabel or r"$^\circ K$"
        self._time_animation(
            self.fet,
            name,
            [],
            title=title,
            barlabel=barlabel,
            dpi=300,
            cmap="viridis",
            pbar_label="F layer electron temperature",
            **kwargs,
        )
