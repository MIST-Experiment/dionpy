import echaim
import numpy as np
from multiprocessing import shared_memory
import iricore


def nan2zero(arr):
    return np.where(np.isnan(arr), 0, arr)


def parallel_iri(dt, heights, batch_lat, batch_lon, shnm_edens, shnm_etemp, arr_shape, batch_i, iriversion):
    shm_edens = shared_memory.SharedMemory(name=shnm_edens)
    shm_etemp = shared_memory.SharedMemory(name=shnm_etemp)
    shedens = np.ndarray(arr_shape, dtype=np.float32, buffer=shm_edens.buf)
    shetemp = np.ndarray(arr_shape, dtype=np.float32, buffer=shm_etemp.buf)
    res = iricore.iri(dt, heights, batch_lat, batch_lon, version=iriversion)

    shedens[batch_i:batch_i + len(batch_lat)] = nan2zero(res.edens)
    shetemp[batch_i:batch_i + len(batch_lat)] = nan2zero(res.etemp)


def parallel_echaim(batch_lat, batch_lon, heights, dt, shnm_edens, arr_shape, batch_i, *args, **kwargs):
    shm_edens = shared_memory.SharedMemory(name=shnm_edens)
    shedens = np.ndarray(arr_shape, dtype=np.float32, buffer=shm_edens.buf)
    shedens[batch_i:batch_i + len(batch_lat)] = echaim.density_profile(batch_lat, batch_lon, heights, dt, *args,
                                                                       **kwargs)
