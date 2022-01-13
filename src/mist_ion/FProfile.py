from .IonModel import *
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

        self.lat_vals = np.linspace(-90., 90., gridsize)
        self.lon_vals = np.linspace(-180., 180., gridsize)
        self.lat = np.repeat(self.lat_vals, gridsize)
        self.lon = np.tile(self.lon_vals, gridsize)
        self.npoints = gridsize * gridsize
        self.gridsize = gridsize

        self.nflayers = nlayers
        self.f_bot = f_bot
        self.f_top = f_top

        self.f_heights = np.linspace(f_bot, f_top, nlayers)

        self.f_e_density = np.empty((gridsize**2, nlayers))

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
                flayer = list(tqdm(pool.imap(
                    _calc_fed_point_star,
                    zip(
                        self.lat,
                        self.lon,
                        it.repeat(self.f_heights[i]),
                        it.repeat(self.dt),
                    )),
                    total=self.npoints,
                    disable=not progressbar,
                    desc=f'Sublayer {i+1}/{self.nflayers}',
                ))
                self.f_e_density[:, i] = np.array([f for f in flayer])


    def save(self, name=None, dir=None):
        """
        # TODO
        """
        if dir == None:
            dir = 'calc_results/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is None:
            name = dir + f"{self.dt.year}_{self.dt.month}_{self.dt.day}_{self.dt.hour}_{self.dt.minute}"
        else:
            name = dir + name

        with h5py.File(name + ".h5", mode='w') as file:
            meta = file.create_dataset('meta', shape=(0,))
            meta.attrs['freq'] = self.freq
            meta.attrs['dt'] = self.dt.strftime('%Y-%m-%d %H:%M')
            meta.attrs['gridsize'] = self.gridsize
            meta.attrs['nflayers'] = self.nflayers
            meta.attrs['f_top'] = self.f_top
            meta.attrs['f_bot'] = self.f_bot

            try:
                file.create_dataset('f_e_density', data=self.f_e_density)
            except TypeError:
                pass

    @classmethod
    def load(cls, filename: str):
        with h5py.File(filename, mode='r') as file:
            meta = file.get('meta')
            obj = cls(
                nlayers=meta.attrs['nflayers'],
                dt=datetime.strptime(meta.attrs['dt'], '%Y-%m-%d %H:%M'),
                freq=meta.attrs['freq'],
                gridsize=meta.attrs['gridsize'],
                f_bot=meta.attrs['f_bot'],
                f_top=meta.attrs['f_top'],
            )
            obj.f_e_density = np.array(file.get('f_e_density'))
        return obj

    def plot(self, data='f_e_density', title=None, label=None, cblim=None, file=None, dpi=300, cmap='viridis'):
        if data == 'f_e_density':
            aver_fe = self.f_e_density.mean(axis=1)
            if title is None:
                title = r'Average $e^-$ density in F layer'
            if label is None:
                label = r"$m^{-3}$"
            fig = plt.figure(figsize=(10, 8))
            m = Basemap(projection='cyl', resolution='l',
                        llcrnrlat=-90, urcrnrlat=90,
                        llcrnrlon=-180, urcrnrlon=180
                        )
            m.drawparallels(np.arange(-90, 91, 20), labels=[1, 0, 0, 0])
            m.drawmeridians(np.arange(-180, 179, 40), labels=[0, 0, 0, 1])
            m.shadedrelief(scale=0.5)
            if cblim is None:
                cblim = (np.min(aver_fe), np.max(aver_fe))
            img = m.pcolormesh(self.lon_vals, self.lat_vals,
                         aver_fe.reshape((self.gridsize, self.gridsize)),
                         latlon=True, cmap=cmap, shading='auto',
                         clim=cblim)
            m.drawcoastlines(color='lightgray', linewidth=0.5)


            plt.title(title)
            plt.xlabel(datetime.strftime(self.dt, '%Y-%m-%d %H:%M'), labelpad=20)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(img, cax=cax).set_label(r'' + label)
            if file is not None:
                plt.savefig(file, dpi=dpi)
                plt.close(fig)
                return
            plt.show()
