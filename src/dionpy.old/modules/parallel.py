import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
from scipy.interpolate import interp1d


def create_shared_block(array):
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    np_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np_array[:] = array[:]
    return shm, np_array


def shared_array(array):
    """
    Returns a copy of array in shared memory that may be used in different processes.
    """
    sharr_base = mp.Array(ctypes.c_double, int(np.prod(array.shape)))
    sharr = np.ctypeslib.as_array(sharr_base.get_obj())
    sharr[:] = array.ravel()[:]
    sharr = sharr.reshape(array.shape)
    return sharr


def interp_val(data1, data2, dt1, dt2, dt):
    """
    Linear interpolation of value(s) between two data points/arrays given their datetimes.
    """
    if dt1 == dt2:
        return data1

    x = np.asarray([0, (dt2 - dt1).total_seconds()])
    y = np.asarray([data1, data2])
    linmod = interp1d(x, y, axis=0)
    x_in = (dt - dt1).total_seconds()
    return linmod(x_in)


def calc_interp_val(el, az, funcs, dts, *args, **kwargs):
    """
    First calculate data from provided functions, then perform an interpolation.
    """
    data1 = funcs[0](el, az, *args, **kwargs)
    data2 = funcs[1](el, az, *args, **kwargs)
    return interp_val(data1, data2, *dts)


def calc_interp_val_star(pars):
    return calc_interp_val(*pars[:-2], *pars[-2], **pars[-1])