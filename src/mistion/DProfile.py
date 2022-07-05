import matplotlib.pyplot as plt
import numpy as np

from .IonModel import *
from scipy.interpolate import interp2d, interp1d
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _calc_ded_points(lats, lons, alts, dt):
    f_alt_prof = iricore.IRI(dt, alts, lats, lons, replace_missing=0)
    return f_alt_prof["ne"]


def _calc_ded_points_star(pars):
    return _calc_fed_points(*pars)


class DProfile:
    """
    Parameters
    ----------
    nlayers : int
            Number of layers in F-layer
    d_bot : float
        Lower limit of the F-layer in meters
    d_top : float
        Upper limit of the F-layer in meters
    freq : float
        Frequency in Hz of signal at which all model values will be calculated
    dt : datetime
        Date and time of observation in format "yyyy-mm-dd hh:mm"
    gridsize : int, optional
            Resolution of the coordinate grid. The total number of points will be [gridsize x gridsize].
    """

    def __init__(self, nlayers, dt, freq, gridsize=100, d_bot=60, d_top=90):
        self.freq = freq
        self.dt = dt

        self.lat_vals = np.linspace(-90.0, 90.0, gridsize)
        self.lon_vals = np.linspace(-180.0, 180.0, gridsize)
        self.lat = np.repeat(self.lat_vals, gridsize)
        self.lon = np.tile(self.lon_vals, gridsize)
        self.npoints = gridsize * gridsize
        self.gridsize = gridsize

        self.ndlayers = nlayers
        self.d_bot = d_bot
        self.d_top = d_top

        self.d_heights = np.linspace(d_bot, d_top, nlayers)

        self.d_e_density = np.empty((gridsize**2, nlayers))
        self.interp_layers = [None for _ in range(self.ndlayers)]

    def _interpolate_layers(self):
        for i in range(self.ndlayers):
            self.interp_layers[i] = interp2d(
                self.lon_vals,
                self.lat_vals,
                self.d_e_density[:, i].reshape(self.gridsize, self.gridsize),
                kind="linear",
            )

    def calc(self, processes=1, progressbar=False):
        """
        Calculates d_e_density for all sublayers at all points
        """
        cpus = cpu_count()
        if cpus < processes:
            processes = cpus
            warnings.warn(
                f"You have only {cpus} cpu threads available. Setting number of processes to {cpus}.",
                RuntimeWarning,
                stacklevel=2,
            )
        step = (self.d_top - self.d_bot) / self.ndlayers
        self.d_e_density = _calc_ded_points(
            self.lat, self.lon, [self.d_bot, self.d_top - 1, step], self.dt
        )
        # Not paralellized yet
        # with Pool(processes=processes) as pool:
        #     for i in range(self.nflayers):
        #         flayer = list(tqdm(pool.imap(
        #             _calc_fed_points_star,
        #             zip(
        #                 self.lat,
        #                 self.lon,
        #                 it.repeat(self.f_heights[i]),
        #                 it.repeat(self.dt),
        #             )),
        #             total=self.npoints,
        #             disable=not progressbar,
        #             desc=f'Sublayer {i + 1}/{self.nflayers}',
        #         ))
        #         self.f_e_density[:, i] = np.array([f for f in flayer])

        self._interpolate_layers()

    def save(self, name=None, dir=None):
        """
        # TODO
        """
        import h5py

        if dir is None:
            dir = "fprofile_results/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is None:
            name = os.path.join(
                dir,
                f"{self.dt.year}_{self.dt.month}_{self.dt.day}_{self.dt.hour}_{self.dt.minute}",
            )
        else:
            name = os.path.join(dir, name)

        if not name.endswith(".h5"):
            name += ".h5"

        with h5py.File(name, mode="w") as file:
            meta = file.create_dataset("meta", shape=(0,))
            meta.attrs["freq"] = self.freq
            meta.attrs["dt"] = self.dt.strftime("%Y-%m-%d %H:%M")
            meta.attrs["gridsize"] = self.gridsize
            meta.attrs["ndlayers"] = self.nflayers
            meta.attrs["d_top"] = self.d_top
            meta.attrs["d_bot"] = self.d_bot

            try:
                file.create_dataset("f_e_density", data=self.d_e_density)
            except TypeError:
                pass

    @classmethod
    def load(cls, filename: str):
        import h5py

        if not filename.endswith(".h5"):
            filename += ".h5"
        with h5py.File(filename, mode="r") as file:
            meta = file.get("meta")
            obj = cls(
                nlayers=meta.attrs["ndlayers"],
                dt=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
                freq=meta.attrs["freq"],
                gridsize=meta.attrs["gridsize"],
                d_bot=meta.attrs["d_bot"],
                d_top=meta.attrs["d_top"],
            )
            obj.d_e_density = np.array(file.get("d_e_density"))
            obj._interpolate_layers()
        return obj

    def __call__(self, lat, lon, alt):
        check_latlon(lat, lon)

        if alt < self.d_bot:
            warnings.warn(
                f"Entered alt = {alt} value is out of calculated limits. Setting alt of {self.d_bot}.",
                RuntimeWarning,
                stacklevel=2,
            )
            alt = self.d_bot
        if alt > self.d_top:
            warnings.warn(
                f"Entered alt = {alt} value is out of calculated limits. Setting alt of {self.d_bot}.",
                RuntimeWarning,
                stacklevel=2,
            )
            alt = self.d_top

        if None in self.interp_layers:
            raise OrderError("First you must calculate or load the model.")
        alt_prof = interp1d(
            self.d_heights,
            [self.interp_layers[i](lon, lat)[0] for i in range(self.ndlayers)],
        )
        return alt_prof(alt)

    def plot_aver_edensity(
        self,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        if dir == None:
            dir = "pictures/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        aver_fe = self.d_e_density.mean(axis=1)
        if title is None:
            title = r"Average $e^-$ density in D layer"
        if label is None:
            label = r"$m^{-3}$"
        fig = plt.figure(figsize=(10, 8))
        m = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=-90,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
        )
        m.drawparallels(np.arange(-90, 91, 20), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180, 179, 40), labels=[0, 0, 0, 1])
        if cblim is None:
            cblim = (np.min(aver_fe), np.max(aver_fe))
        img = m.pcolormesh(
            self.lon_vals,
            self.lat_vals,
            aver_fe.reshape((self.gridsize, self.gridsize)),
            latlon=True,
            cmap=cmap,
            shading="auto",
            clim=cblim,
        )
        m.drawcoastlines(color="black", linewidth=0.5)

        plt.title(title)
        plt.xlabel(datetime.strftime(self.dt, "%Y-%m-%d %H:%M"), labelpad=20)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(img, cax=cax).set_label(r"" + label)
        if file is not None:
            plt.savefig(os.path.join(dir, file), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return
        return fig

    def plot_interpolated(
        self,
        layer=None,
        gridsize=1000,
        title=None,
        label=None,
        cblim=None,
        file=None,
        dir=None,
        dpi=300,
        cmap="viridis",
    ):
        if dir == None:
            dir = "pictures/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        lon_vals = np.linspace(-180, 180, gridsize)
        lat_vals = np.linspace(-90, 90, gridsize)

        if layer is None:
            f = interp2d(
                self.lon_vals,
                self.lat_vals,
                self.d_e_density.mean(axis=1).reshape(self.gridsize, self.gridsize),
                kind="linear",
            )
            fe = f(lon_vals, lat_vals)
            if title is None:
                title = r"Average $e^-$ density in D layer"
        elif (il := int(layer)) >= 1 and il <= self.ndlayers:
            f = self.interp_layers[il - 1]
            if f is None:
                raise OrderError("First you must calculate or load the model.")
            else:
                fe = f(lon_vals, lat_vals)
                if title is None:
                    title = r"$e^-$ density in F layer " + f"({il} sublayer)"
        else:
            raise ValueError(
                "The 'layer' parameter must be either None or int = [1, nlayers]"
            )

        if label is None:
            label = r"$m^{-3}$"
        fig = plt.figure(figsize=(10, 8))
        map = Basemap(
            projection="cyl",
            resolution="l",
            llcrnrlat=-90,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
        )
        map.drawparallels(np.arange(-90, 91, 20), labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(-180, 179, 40), labels=[0, 0, 0, 1])
        if cblim is None:
            cblim = (np.min(fe), np.max(fe))

        img = map.pcolormesh(
            lon_vals,
            lat_vals,
            fe,
            latlon=True,
            cmap=cmap,
            shading="auto",
            clim=cblim,
        )
        map.drawcoastlines(color="black", linewidth=0.5)

        plt.title(title)
        plt.xlabel(datetime.strftime(self.dt, "%Y-%m-%d %H:%M"), labelpad=20)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(img, cax=cax).set_label(r"" + label)
        if file is not None:
            plt.savefig(os.path.join(dir, file), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return
        return fig
