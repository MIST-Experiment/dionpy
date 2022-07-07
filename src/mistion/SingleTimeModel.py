import os
from datetime import datetime

import numpy as np
import healpy as hp

from .DLayer import DLayer
from .FLayer import FLayer
from .modules.helpers import none_or_array, elaz_mesh
from .modules.ion_tools import trop_refr
from typing import Tuple, Union


class SingleTimeModel:
    def __init__(
        self,
        dt: datetime,
        position: Tuple[float, float, float],
        nside: int = 128,
        dbot: float = 60,
        dtop: float = 90,
        ndlayers: int = 10,
        fbot: float = 60,
        ftop: float = 90,
        nflayers: int = 30,
    ):
        if isinstance(dt, datetime):
            self.dt = dt
        else:
            raise ValueError("Parameter dt must be a datetime object.")
        self.pos = position
        self.nside = nside
        self.dlayer = DLayer(
            dt, position, dbot, dtop, ndlayers, nside
        )
        self.flayer = FLayer(
            dt, position, fbot, ftop, nflayers, nside
        )

    def _check_elaz(self, el, az, size_err=True):
        if el is None or az is None:
            el = np.linspace(*self.elrange, self.gridsize)
            az = np.linspace(*self.azrange, self.gridsize)
        else:
            el = np.asarray(el)
            az = np.asarray(az)
            if el.size != az.size and size_err:
                raise ValueError("Elevation and azimuth must have the same size")
        return el, az

    def troprefr(self, el=None, az=None):
        """
        Refraction in the troposphere in degrees. Depends only on elevation angle; second argument exists to make
        this method similar to others, but is not required.
        """
        if az is not None:
            el, az = self._check_elaz(el, az, size_err=False)
            el_rows, az_rows = np.meshgrid(el, az)
            return trop_refr(np.deg2rad(90 - el_rows))
        return trop_refr(np.deg2rad(90 - el))

    def frefr(self, el, az):
        """
        Calculates refraction in F layer for a given model of ionosphere. Output is the change of zenith angle theta
        (theta -> theta + dtheta). If coordinates are floats the output will be a single number; if they are arrays -
        the output will be a 2D array with dimensions az.size x el.size (according to np.meshgrid(el, az)).

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
        return self.flayer.frefr(el, az)

    def datten(self, el, az, col_freq="default", troposphere=True):
        """
                Calculates attenuation in D layer for a given model of ionosphere. Output is the attenuation factor between 0
                (total attenuation) and 1 (no attenuation). If coordinates are floats the output will be a single number; if
                they are arrays - the output will be a 2D array with dimensions el.size x az.size.

                Parameters
                ----------
                el : None | float | np.ndarray
                    Elevation of observation(s). If not provided - the model's grid will be used.
                az : None | float | np.ndarray
                    Azimuth of observation(s). If not provided - the model's grid will be used.
                col_freq : str, float
                    The collision frequency ('default', 'nicolet', 'setty', 'aggrawal', or float in Hz)
                troposhpere : Bool, default=True
                    Account for troposphere refraction bias

                Returns
                -------
                np.ndarray
        """
        return self.dlayer.datten(el, az, self.freq, col_freq, troposphere)

    def save(self, dir=None, name=None):
        import h5py

        if dir == None:
            dir = "ion_models/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is None:
            name = os.path.join(
                dir,
                f"{self.dt.year:04d}{self.dt.month:02d}{self.dt.day:02d}{self.dt.hour:02d}{self.dt.minute:02d}",
            )
        else:
            name = os.path.join(dir, name)

        if not name.endswith(".h5"):
            name += ".h5"

        with h5py.File(name, mode="w") as file:
            meta = file.create_dataset("meta", shape=(0,))
            meta.attrs["lat"] = self.pos[0]
            meta.attrs["lon"] = self.pos[1]
            meta.attrs["alt"] = self.pos[2]
            meta.attrs["freq"] = self.freq
            meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
            meta.attrs["gridsize"] = self.gridsize

            meta.attrs["ndlayers"] = self.dlayer.ndlayers
            meta.attrs["dtop"] = self.dlayer.dtop
            meta.attrs["dbot"] = self.dlayer.dbot

            meta.attrs["nflayers"] = self.flayer.nflayers
            meta.attrs["ftop"] = self.flayer.fbot
            meta.attrs["fbot"] = self.flayer.ftop

            meta.attrs["elstart"] = self.elrange[0]
            meta.attrs["elend"] = self.elrange[1]
            meta.attrs["azstart"] = self.azrange[0]
            meta.attrs["azend"] = self.azrange[1]

            if self.dlayer._d_e_density is not None:
                file.create_dataset("d_e_density", data=self.dlayer._d_e_density)
                file.create_dataset("d_e_temp", data=self.dlayer._d_e_temp)

            if self.flayer._f_e_density is not None:
                file.create_dataset("f_e_density", data=self.flayer._f_e_density)
                file.create_dataset("f_e_temp", data=self.flayer._f_e_temp)
                file.create_dataset("dtheta", data=self.flayer._dtheta)

    @classmethod
    def load(cls, path: str):
        import h5py

        if not path.endswith(".h5"):
            path += ".h5"
        with h5py.File(path, mode="r") as file:
            meta = file.get("meta")
            obj = cls(
                datetime=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
                position=(
                    meta.attrs["lat"],
                    meta.attrs["lon"],
                    meta.attrs["alt"],
                ),
                freq=meta.attrs["freq"],
                gridsize=meta.attrs["gridsize"],
                elrange=(meta.attrs["elstart"], meta.attrs["elend"]),
                azrange=(meta.attrs["azstart"], meta.attrs["azend"]),
                dbot=meta.attrs["dbot"],
                dtop=meta.attrs["dtop"],
                ndlayers=meta.attrs["ndlayers"],
                fbot=meta.attrs["fbot"],
                ftop=meta.attrs["ftop"],
                nflayers=meta.attrs["nflayers"],
            )

            obj.dlayer._d_e_density = none_or_array(file.get("d_e_density"))
            obj.dlayer._d_e_temp = none_or_array(file.get("d_e_temp"))

            if obj.dlayer._d_e_density is not None:
                obj.dlayer._interpolate_d_layer()

            obj.flayer._f_e_density = none_or_array(file.get("f_e_density"))
            obj.flayer._f_e_temp = none_or_array(file.get("f_e_temp"))
            obj.flayer._dtheta = none_or_array(file.get("dtheta"))

            if obj.flayer._f_e_density is not None:
                obj.flayer._interpolate_f_layer()

        return obj

    def _polar_plot(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        title=None,
        barlabel=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        import matplotlib.pyplot as plt
        plotlabel = plotlabel or "UTC time: " + datetime.strftime(
            self.dt, "%Y-%m-%d %H:%M"
        )
        cblim = cblim or (np.min(data[2]), np.max(data[2]))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        img = ax.pcolormesh(
            data[0],
            data[1],
            data[2],
            cmap=cmap,
            vmin=cblim[0],
            vmax=cblim[1],
            shading="auto",
        )
        ax.set_rticks([90, 60, 30, 0])
        ax.scatter(0, 0, c="red", s=5)
        ax.set_theta_zero_location("S")
        plt.colorbar(img).set_label(barlabel)
        plt.title(title)
        plt.xlabel(plotlabel)

        if saveto is not None:
            head, tail = os.path.split(saveto)
            if not os.path.exists(head):
                os.makedirs(head)
            plt.savefig(saveto, dpi=dpi)
            plt.close(fig)
            return
        return fig

    def plot_ded(
        self,
        gridsize=100,
        layer=None,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        barlabel = r"$m^-3$"
        if title is None:
            if layer is None:
                title = r"Average $n_e$ in the D layer"
            else:
                title = r"$n_e$ " + f"in the {layer} sublayer of the D layer"
        el, az = elaz_mesh(gridsize)
        ded = self.dlayer.ded(el, az, layer)
        return self._polar_plot(
            (np.deg2rad(az), 90-el, ded),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_det(
        self,
        gridsize=100,
        layer=None,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        barlabel = r"$K^\circ$"
        if title is None:
            if layer is None:
                title = r"Average $T_e$ in the D layer"
            else:
                title = r"$T_e$ " + f"in the {layer} sublayer of the D layer"
        el, az = elaz_mesh(gridsize)
        ded = self.dlayer.det(el, az, layer)
        return self._polar_plot(
            (np.deg2rad(az), 90 - el, ded),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_fed(
        self,
        interpolated=True,
        layer=None,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        barlabel = r"$m^-3$"
        gs = 1000 if interpolated else self.gridsize
        az_vals, az_rows, el_vals, el_rows = generate_plot_grid(
            *self.elrange, *self.azrange, gs
        )
        if layer is not None:
            if not 0 < layer <= self.flayer.nflayers:
                raise ValueError("The layer option must be 0 < layer < nlayers.")
            fed = self.flayer._interp_fed[layer - 1](el_vals, az_vals)
            title = (
                title
                or r"$n_e$ in the "
                + f"{layer}/{self.flayer.nflayers} sublayer of the F layer"
            )
        else:
            fed = self.flayer._interp_feda(el_vals, az_vals)
            title = title or r"Average $n_e$ in the F layer"
        return self._polar_plot(
            (np.deg2rad(az_rows), 90 - el_rows, fed),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_fet(
        self,
        interpolated=True,
        layer=None,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        barlabel = r"$K^\circ$"
        gs = 1000 if interpolated else self.gridsize
        az_vals, az_rows, el_vals, el_rows = generate_plot_grid(
            *self.elrange, *self.azrange, gs
        )
        if layer is not None:
            if not 0 < layer <= self.flayer.nflayers:
                raise ValueError("The layer option must be 0 < layer < nlayers.")
            fet = self.flayer._interp_fet[layer - 1](el_vals, az_vals)
            title = (
                title
                or r"$T_e$ in the "
                + f"{layer}/{self.dlayer.ndlayers} sublayer of the F layer"
            )
        else:
            fet = self.flayer._interp_feta(el_vals, az_vals)
            title = title or r"Average $T_e$ in the F layer"
        return self._polar_plot(
            (np.deg2rad(az_rows), 90 - el_rows, fet),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_datten(
        self,
        freq,
        troposphere=True,
        gridsize=100,
        title=None,
        plotlabel=None,
        barlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap="viridis",
    ):
        el, az = elaz_mesh(gridsize)
        datten = self.dlayer.datten(el, az, freq, troposphere=troposphere)
        title = title or r"Average $f_{atten}$ in the D layer at " + f"{freq/1e6:.1f} MHz"
        return self._polar_plot(
            (np.deg2rad(az), 90-el, datten),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_frefr(
        self,
        interpolated=True,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap='viridis_r',
    ):
        barlabel = r"$\delta \theta$, deg"
        gs = 1000 if interpolated else self.gridsize
        az_vals, az_rows, el_vals, el_rows = generate_plot_grid(
            *self.elrange, *self.azrange, gs
        )
        frefr = self.flayer.frefr(el_vals, az_vals)
        title = title or r"Average $f_{atten}$ in the D layer at " + f"{self.freq/1e6} MHz"
        return self._polar_plot(
            (np.deg2rad(az_rows), 90 - el_rows, frefr),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )

    def plot_troprefr(
        self,
        interpolated=True,
        title=None,
        plotlabel=None,
        cblim=None,
        saveto=None,
        dpi=300,
        cmap='viridis_r',
    ):
        barlabel = r"$\delta \theta$, deg"
        gs = 1000 if interpolated else self.gridsize
        az_vals, az_rows, el_vals, el_rows = generate_plot_grid(
            *self.elrange, *self.azrange, gs
        )
        troprefr = self.troprefr(el_vals, az_vals)
        title = title or r"$\delta \theta$ in the troposphere"
        return self._polar_plot(
            (np.deg2rad(az_rows), 90 - el_rows, troprefr),
            title,
            barlabel,
            plotlabel,
            cblim,
            saveto,
            dpi,
            cmap,
        )
