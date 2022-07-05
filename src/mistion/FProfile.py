import matplotlib.pyplot as plt
import numpy as np

from .IonModel import *
from scipy.interpolate import interp2d, interp1d
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _calc_fed_point(lat, lon, alt, dt):
    f_alt_prof = ion.IRI(dt, [alt / 1e3, alt / 1e3, 1], lat, lon)
    return f_alt_prof.ne.data[0]


def _calc_fed_point_star(pars):
    return _calc_fed_point(*pars)


class FProfile:
    """
    Parameters
    ----------
    nlayers : int
            Number of layers in F-layer
    f_bot : float
        Lower limit of the F-layer in meters
    f_top : float
        Upper limit of the F-layer in meters
    freq : float
        Frequency in Hz of signal at which all model values will be calculated
    dt : datetime
        Date and time of observation in format "yyyy-mm-dd hh:mm"
    gridsize : int, optional
            Resolution of the coordinate grid. The total number of points will be [gridsize x gridsize].
    """

    def __init__(self, nlayers, dt, freq, gridsize=100, f_bot=1.5e5, f_top=5e5):
        # if not -90 <= lat0 <= 90:
        #     raise ValueError("Latitude of the instrument must be in range [-90, 90]")
        # if not -180 <= lon0 < 180:
        #     raise ValueError("Longitude of the instrument must be in range [-180, 180]")
        #
        # self.lat0 = lat0
        # self.lon0 = lon0
        # self.alt0 = alt0
        self.freq = freq
        self.dt = dt

        self.lat_vals = np.linspace(-90.0, 90.0, gridsize)
        self.lon_vals = np.linspace(-180.0, 180.0, gridsize)
        self.lat = np.repeat(self.lat_vals, gridsize)
        self.lon = np.tile(self.lon_vals, gridsize)
        self.npoints = gridsize * gridsize
        self.gridsize = gridsize

        self.nflayers = nlayers
        self.f_bot = f_bot
        self.f_top = f_top

        self.f_heights = np.linspace(f_bot, f_top, nlayers)

        self.f_e_density = np.empty((gridsize**2, nlayers))
        self.interp_layers = [None for i in range(self.nflayers)]

    def _interpolate_layers(self):
        for i in range(self.nflayers):
            self.interp_layers[i] = interp2d(
                self.lon_vals,
                self.lat_vals,
                self.f_e_density[:, i].reshape(self.gridsize, self.gridsize),
                kind="linear",
            )

    def calc(self, processes=1, progressbar=False):
        """
        Calculates f_e_density for all sublayers at all points
        """
        cpus = cpu_count()
        if cpus < processes:
            processes = cpus
            warnings.warn(
                f"You have only {cpus} cpu threads available. Setting number of processes to {cpus}.",
                RuntimeWarning,
                stacklevel=2,
            )

        with Pool(processes=processes) as pool:
            for i in range(self.nflayers):
                flayer = list(
                    tqdm(
                        pool.imap(
                            _calc_fed_point_star,
                            zip(
                                self.lat,
                                self.lon,
                                it.repeat(self.f_heights[i]),
                                it.repeat(self.dt),
                            ),
                        ),
                        total=self.npoints,
                        disable=not progressbar,
                        desc=f"Sublayer {i + 1}/{self.nflayers}",
                    )
                )
                self.f_e_density[:, i] = np.array([f for f in flayer])

        self._interpolate_layers()

    def save(self, name=None, dir=None):
        """
        # TODO
        """
        import h5py

        if dir == None:
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
            meta.attrs["nflayers"] = self.nflayers
            meta.attrs["f_top"] = self.f_top
            meta.attrs["f_bot"] = self.f_bot

            try:
                file.create_dataset("f_e_density", data=self.f_e_density)
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
                nlayers=meta.attrs["nflayers"],
                dt=datetime.strptime(meta.attrs["dt"], "%Y-%m-%d %H:%M"),
                freq=meta.attrs["freq"],
                gridsize=meta.attrs["gridsize"],
                f_bot=meta.attrs["f_bot"],
                f_top=meta.attrs["f_top"],
            )
            obj.f_e_density = np.array(file.get("f_e_density"))
            obj._interpolate_layers()
        return obj

    def __call__(self, lat, lon, alt):
        check_latlon(lat, lon)

        if alt < self.f_bot:
            warnings.warn(
                f"Entered alt = {alt} value is out of calculated limits. Setting alt of {self.f_bot}.",
                RuntimeWarning,
                stacklevel=2,
            )
            alt = self.f_bot
        if alt > self.f_top:
            warnings.warn(
                f"Entered alt = {alt} value is out of calculated limits. Setting alt of {self.f_bot}.",
                RuntimeWarning,
                stacklevel=2,
            )
            alt = self.f_top

        if None in self.interp_layers:
            raise OrderError("First you must calculate or load the model.")
        alt_prof = interp1d(
            self.f_heights,
            [self.interp_layers[i](lon, lat)[0] for i in range(self.nflayers)],
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
        aver_fe = self.f_e_density.mean(axis=1)
        if title is None:
            title = r"Average $e^-$ density in F layer"
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
                self.f_e_density.mean(axis=1).reshape(self.gridsize, self.gridsize),
                kind="linear",
            )
            fe = f(lon_vals, lat_vals)
            if title is None:
                title = r"Average $e^-$ density in F layer"
        elif (il := int(layer)) >= 1 and il <= self.nflayers:
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

    def track_ray(self, lat: float, lon: float, alt: float, az: float, el: float):
        """
        Parameters
        ----------
        lat : float
            Latitude of the instrument in degrees
        lon : float
            Longitude of the instrument in degrees
        alt : float
            Altitude of the instrument in meters
        az : azimuth of the observation
            Frequency in Hz of signal at which all model values will be calculated
        el : elevation of the observation
            Date and time of observation in format "yyyy-mm-dd hh:mm"
        """
        check_latlon(lat, lon)
        R_E = 6371000.0
        ns = np.empty(self.nflayers)  # refractive indices
        lats = np.empty(self.nflayers + 1)  # refractive indices
        lons = np.empty(self.nflayers + 1)  # refractive indices
        els = np.empty(self.nflayers + 1)  # refractive indices
        heights = np.empty(self.nflayers + 1)
        f_e_density = np.empty(self.nflayers)
        phis = np.empty(self.nflayers)  # angles of refraction
        delta_phi = 0.0  # total change in angle

        lats[0] = lat
        lons[0] = lon
        heights[0] = alt
        els[0] = el

        # Distance from telescope to first layer
        r_slant = srange((90.0 - el) * np.pi / 180.0, self.f_heights[0] - alt)
        # Geodetic coordinates of 'hit point' on the first layer
        lat_ray, lon_ray, h_ray = pm.aer2geodetic(az, el, r_slant, lat, lon, alt)
        lats[1] = lat_ray
        lons[1] = lon_ray
        heights[1] = h_ray
        # The sides of the 1st triangle
        d_tel = R_E + alt  # Distance from Earth center to telescope
        d_cur = R_E + h_ray  # Distance from Earth center to layer

        # The inclination angle at the 1st interface using law of cosines [rad]
        cosphi_inc = (r_slant**2 + d_cur**2 - d_tel**2) / (2 * r_slant * d_cur)
        assert cosphi_inc <= 1, "Something is wrong with coordinates."
        phi_inc = np.arccos(cosphi_inc)

        # Refraction index of air
        n_cur = 1.0

        # Get IRI info of point
        # f_alt_prof = ion.IRI(dt, [h_cur / 1e3, h_cur / 1e3, 1], lat, lon)
        f_e_density[0] = self.__call__(lat_ray, lon_ray, h_ray)

        # Refraction index of 1st point
        n_next = n_f(f_e_density[0], self.freq)
        ns[0] = n_next

        # The outgoing angle at the 1st interface using Snell's law
        phi_ref = refr_angle(n_cur, n_next, phi_inc)
        phis[0] = phi_ref
        delta_phi += phi_ref - phi_inc

        # el_cur = el - (phi_ref - phi_inc)
        el_cur = np.rad2deg(np.pi / 2 - phi_ref)
        els[1] = el_cur

        n_cur = n_next

        for i in range(1, self.nflayers):
            h_next = self.f_heights[i]
            d_next = R_E + h_next

            # Angle between d_cur and r_slant
            int_angle = np.pi - phi_ref
            # The inclination angle at the i-th interface using law of sines [rad]
            phi_inc = np.arcsin(np.sin(int_angle) * d_cur / d_next)

            # Getting r2 using law of cosines
            r_slant = srange((90.0 - el_cur) * np.pi / 180.0, d_next - d_cur)
            # r_slant = d_cur * np.cos(int_angle) + np.sqrt(d_next ** 2 - d_cur ** 2 * np.sin(int_angle) ** 2)

            # Get geodetic coordinates of point
            lat_ray, lon_ray, h_ray = pm.aer2geodetic(
                az, el_cur, r_slant, lat_ray, lon_ray, h_ray
            )
            lats[i + 1] = lat_ray
            lons[i + 1] = lon_ray
            heights[i + 1] = h_ray
            # Get IRI info of 2nd point
            f_e_density[i] = self.__call__(lat_ray, lon_ray, h_ray)
            if f_e_density[i] < 0:
                raise ValueError("Something went wrong. Number density cannot be < 0.")

            # Refractive indices
            n_next = n_f(f_e_density[i], self.freq)
            ns[i] = n_next

            # If this is the last point then use refractive index of vacuum
            if i == self.nflayers - 1:
                n_next = 1

            # The outgoing angle at the 2nd interface using Snell's law
            phi_ref = refr_angle(n_cur, n_next, phi_inc)
            phis[i] = phi_ref
            delta_phi += phi_ref - phi_inc

            # Update variables for new interface
            # el_cur = el_cur - (phi_ref - phi_inc)
            el_cur = np.rad2deg(np.pi / 2 - phi_ref)
            els[i + 1] = el_cur
            n_cur = n_next
            d_cur = d_next

        return {
            "n": ns,
            "fedensity": f_e_density,
            "phi": phis,
            "delta_phi": delta_phi,
            "lat": lats,
            "lon": lons,
            "alt": heights,
            "el": els,
        }

    def save_track_report(self, track_report: dict, dir: str = None, dpi: int = 300):
        data = track_report
        if dir == None:
            dir = "track_report/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Ray on cylindrycal map
        lon_vals = np.linspace(-180, 180, 1000)
        lat_vals = np.linspace(-90, 90, 1000)
        f = interp2d(
            self.lon_vals,
            self.lat_vals,
            self.f_e_density.mean(axis=1).reshape(self.gridsize, self.gridsize),
            kind="linear",
        )
        fe = f(lon_vals, lat_vals)
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
        cblim = (np.min(fe), np.max(fe))
        img = map.pcolormesh(
            lon_vals,
            lat_vals,
            fe,
            latlon=True,
            cmap="viridis",
            shading="auto",
            clim=cblim,
        )
        map.drawcoastlines(color="black", linewidth=0.5)
        x, y = map(data["lon"], data["lat"])
        map.plot(
            x[:2],
            y[:2],
            marker=None,
            c="r",
            linestyle=":",
            label="Telescope -> 1st layer",
        )
        map.plot(x[1:], y[1:], marker=None, c="r", label="Inside ionosphere")
        plt.legend()
        plt.title("Latitude-longitude ray track")

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(img, cax=cax)
        plt.savefig(os.path.join(dir, "latlon.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # lat - alt
        plt.plot(data["lat"], data["alt"], c="r", label="Ray track")
        plt.axhline(
            y=self.f_heights[0], linestyle="-.", c="k", lw="0.5", label="F layer"
        )
        for h in self.f_heights[1:]:
            plt.axhline(y=h, linestyle="-.", c="k", lw="0.5")
        plt.legend()
        plt.xlabel(r"Latitude, $^{\circ}$")
        plt.ylabel(r"Altitude, $m$")
        plt.title("Latitude-altitude ray track")
        plt.savefig(os.path.join(dir, "latalt.png"), dpi=dpi, bbox_inches="tight")
        plt.clf()

        # lon - alt
        plt.plot(data["lon"], data["alt"], c="r", label="Ray track")
        plt.axhline(
            y=self.f_heights[0], linestyle="-.", c="k", lw="0.5", label="F layer"
        )
        for h in self.f_heights[1:]:
            plt.axhline(y=h, linestyle="-.", c="k", lw="0.5")
        plt.legend()
        plt.xlabel(r"Longitude, $^{\circ}$")
        plt.ylabel(r"Altitude, $m$")
        plt.title("Longitude-altitude ray track")
        plt.savefig(os.path.join(dir, "lonalt.png"), dpi=dpi, bbox_inches="tight")
        plt.clf()

        return
