import itertools
from multiprocessing import cpu_count, Pool

import iricore
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def iri_star(pars):
    return iricore.IRI(*pars)


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


def interp_val_star(pars):
    return interp_val(*pars)


def calc_refatt(func, el, az, freq, **kwargs):
    """
    Outside class function to make calculation of attenuation and refraction possible in parallel.
    """
    return func(el, az, freq, **kwargs)


def calc_refatt_star(pars):
    return calc_refatt(*pars[:-1], **pars[-1])


def calc_refatt_par(func, el, az, freqs, pbar_desc, **kwargs):
    """
    Implements parallel calculation of refraction/attenuation.
    """
    nproc = np.min([cpu_count(), len(freqs)])
    disable = True if pbar_desc is None else False
    rep_kwargs = [kwargs for _ in range(len(freqs))]
    with Pool(processes=nproc) as pool:
        res = list(
            tqdm(
                pool.imap(
                    calc_refatt_star,
                    zip(
                        itertools.repeat(func),
                        itertools.repeat(el),
                        itertools.repeat(az),
                        freqs,
                        rep_kwargs,
                    ),
                ),
                total=len(freqs),
                disable=disable,
                desc=pbar_desc,
            )
        )
    return np.array(res)


def calc_interp_val(el, az, funcs, dts, *args, **kwargs):
    """
    First calculate data from provided functions, then perform an interpolation.
    """
    data1 = funcs[0](el, az, *args, **kwargs)
    data2 = funcs[1](el, az, *args, **kwargs)
    return interp_val(data1, data2, *dts)


def calc_interp_val_star(pars):
    return calc_interp_val(*pars[:-2], *pars[-2], **pars[-1])


def calc_interp_val_par(el, az, funcs, dts, pbar_desc, *args, **kwargs):
    """
    Implements parallel calculation and interpolation of data at given datetimes.
    """
    nproc = np.min([cpu_count(), len(funcs)])
    disable = True if pbar_desc is None else False
    rep_args = [args for _ in range(len(dts))]
    rep_kwargs = [kwargs for _ in range(len(dts))]
    with Pool(processes=nproc) as pool:
        res = list(
            tqdm(
                pool.imap(
                    calc_interp_val_star,
                    zip(
                        itertools.repeat(el),
                        itertools.repeat(az),
                        funcs,
                        dts,
                        rep_args,
                        rep_kwargs,
                    ),
                ),
                total=len(dts),
                disable=disable,
                desc=pbar_desc,
            )
        )
    return np.array(res)
