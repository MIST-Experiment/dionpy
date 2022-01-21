from .IonModel import IonModel, OrderError, _d_temp_density, _calc_flayer
from tqdm import tqdm
import numpy as np
import warnings
# import mpi4py.rc
# mpi4py.rc.threads = False
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, as_completed
COMM = MPI.COMM_WORLD
MPI_RANK = COMM.Get_rank()
MPI_SIZE = COMM.Get_size()


class IonModelMPI(IonModel):
    def calc(self, progressbar=False, layer=None, **kwargs):
        """
        Calculates all necessary values for ionosphere impact modeling - D-layer [electron density(d_e_density),
        electron temperature(d_e_temp), attenuation factor(d_attenuation), average temperature(d_avg_temp)] and F-layer
        [electron density(f_e_density), angle of the outgoing refracted beam at each layer(phis),
        the net deviation of the elevation angle for each coordinate(delta_phi), refractive index at each layer(ns)].
        """

        if None in [self.ndlayers, self.d_bot, self.d_top] and layer != 'f' and layer != 'F':
            raise OrderError("You have to set up parameters for the D layer first (use the setup_dlayer() method)")

        if None in [self.nflayers, self.f_bot, self.f_top] and layer != 'd' and layer != 'D':
            raise OrderError("You have to set up parameters for the F layer first (use the setup_flayer() method)")

        if MPI_SIZE <= 1:
            warnings.warn(
                "Can't split data into equal number of rows. "
                "Please make sure your coordinates represent a square grid.",
                RuntimeWarning,
                stacklevel=2,
            )
            if layer != 'f' and layer != 'F':
                dlayer = []

                for i in tqdm(range(self.npoints), desc='D-layer', disable=not progressbar, dynamic_ncols=True):
                    dlayer.append(_d_temp_density(
                        self.dt,
                        self.d_bot,
                        self.d_top,
                        self.ndlayers,
                        self.el[i],
                        self.az[i],
                        self.lat0,
                        self.lon0,
                        self.alt0,
                    ))
                self.d_e_density = np.vstack([d[0] for d in dlayer])
                self.d_e_temp = np.vstack([d[1] for d in dlayer])
                self._calc_d_attenuation()
                self._calc_d_avg_temp()

            if layer != 'd' and layer != 'D':
                flayer = []

                for i in tqdm(range(self.npoints), desc='F-layer', disable=not progressbar):
                    flayer.append(_calc_flayer(
                        self.dt,
                        self.f_bot,
                        self.f_top,
                        self.nflayers,
                        self.el[i],
                        self.az[i],
                        self.lat0,
                        self.lon0,
                        self.alt0,
                        self.freq
                    ))
                self.f_e_density = np.vstack([f[0] for f in flayer])
                self.phis = np.vstack([f[1] for f in flayer])
                self.delta_phi = np.vstack([f[2] for f in flayer]).reshape([-1])
                self.ns = np.vstack([f[3] for f in flayer])

        else:
            if layer != 'f' and layer != 'F':
                with MPIPoolExecutor(main=False) as pool:
                    futures = [pool.submit(
                        _d_temp_density,
                        self.dt,
                        self.d_bot,
                        self.d_top,
                        self.ndlayers,
                        self.el[i],
                        self.az[i],
                        self.lat0,
                        self.lon0,
                        self.alt0,
                    ) for i in range(self.npoints)]

                    if progressbar:
                        for _ in tqdm(as_completed(futures), total=len(futures), desc='D layer'):
                            pass

                self.d_e_density = np.vstack([f.result()[0] for f in futures])
                self.d_e_temp = np.vstack([f.result()[1] for f in futures])
                self._calc_d_attenuation()
                self._calc_d_avg_temp()

            if layer != 'd' and layer != 'D':
                with MPIPoolExecutor(main=False) as pool:
                    futures = [pool.submit(
                        _calc_flayer,
                        self.dt,
                        self.f_bot,
                        self.f_top,
                        self.nflayers,
                        self.el[i],
                        self.az[i],
                        self.lat0,
                        self.lon0,
                        self.alt0,
                        self.freq,
                    ) for i in range(self.npoints)]

                    if progressbar:
                        for _ in tqdm(as_completed(futures), total=len(futures), desc='F layer'):
                            pass

                self.f_e_density = np.vstack([f.result()[0] for f in futures])
                self.phis = np.vstack([f.result()[1] for f in futures])
                self.delta_phi = np.vstack([f.result()[2] for f in futures]).reshape([-1])
                self.ns = np.vstack([f.result()[3] for f in futures])


