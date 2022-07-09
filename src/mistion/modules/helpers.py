import os
from datetime import datetime
from typing import Tuple

import iricore
import numpy as np
from scipy.interpolate import interp1d

from .ion_tools import srange
from pymap3d import aer2geodetic
import healpy as hp
import matplotlib


class TextColor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Ellipsoid:
    """
    Custom ellipsoid for pymap3d package. Implements a simple sphere.
    """

    def __init__(self):
        self.semimajor_axis = 6378100.0
        self.semiminor_axis = 6378100.0
        self.flattening = 0.0
        self.thirdflattening = 0.0
        self.eccentricity = 0.0


def none_or_array(vals):
    if vals is None:
        return None
    return np.array(vals)


def polar_plot(
        dt,
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
        dt, "%Y-%m-%d %H:%M"
    )
    cblim = cblim or (np.min(data[2]), np.max(data[2]))

    fig = plt.figure(figsize=(8, 8))
    ax: plt.Axes = fig.add_subplot(111, projection="polar")
    img = ax.pcolormesh(
        data[0],
        data[1],
        data[2],
        cmap=cmap,
        vmin=cblim[0],
        vmax=cblim[1],
        shading="auto",
    )
    ax.grid(color="gray", linestyle=":")
    ax.set_theta_zero_location("S")
    ax.set_rticks([90, 60, 30, 0], Fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelcolor='gray')
    # ax.scatter(0, 0, c="red", s=5)
    plt.colorbar(img, fraction=0.042, pad=0.08).set_label(label=barlabel, size=10)
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(plotlabel, fontsize=10)

    if saveto is not None:
        head, tail = os.path.split(saveto)
        if not os.path.exists(head):
            os.makedirs(head)
        plt.savefig(saveto, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return
    return fig


def check_elaz_shape(el, az):
    if not isinstance(el, float) and not isinstance(az, float):
        if isinstance(el, np.ndarray) and isinstance(el, np.ndarray):
            if not el.shape == az.shape:
                raise ValueError("Elevation and azimuth must be the same length.")
        else:
            raise ValueError(
                "Elevation and azimuth must be either floats or numpy arrays."
            )


def sky2ll(el, az, height, pos):
    """
    Converts visible elevation and azimuth to geographic coordinates with given height of the visible point

    Parameters
    ----------
    el : float | np.ndarray
        Elevation of observation(s) in deg.
    az : float | np.ndarray
        Azimuth of observation(s) in deg.
    height : float
        Height of observable point(s) in km.
    pos: Tuple[float, float, float]
        Geographical coordinates and height in m of the telescope

    Returns
    -------
    obs_lat : float | np.ndarray
        Observable geographical latitude.
    obs_lon : float | np.ndarray
        Observable geographical longitude.
    """
    d_srange = srange(np.deg2rad(90 - el), height * 1e3)
    obs_lat, obs_lon, _ = aer2geodetic(az, el, d_srange, *pos, Ellipsoid())
    return obs_lat, obs_lon


def elaz_mesh(gridsize):
    el = np.linspace(0, 90, gridsize, endpoint=True)
    az = np.linspace(0, 360, gridsize)
    els, azs = np.meshgrid(el, az)
    return els, azs


def eval_layer(
    el, az, nside, position, hbot, htop, nlayers, obs_pixels, data, layer=None
):
    check_elaz_shape(el, az)
    heights = np.linspace(hbot, htop, nlayers)
    map_ = np.zeros(hp.nside2npix(nside)) + hp.UNSEEN
    if layer is None:
        res = np.empty((*el.shape, nlayers))
        for i in range(nlayers):
            map_[obs_pixels] = data[:, i]
            obs_lat, obs_lon = sky2ll(el, az, heights[i], position)
            res[:, :, i] = hp.pixelfunc.get_interp_val(
                map_, obs_lon, obs_lat, lonlat=True
            )
        return res.mean(axis=2)
    elif isinstance(layer, int) and layer < nlayers + 1:
        map_[obs_pixels] = data[:, layer]
        obs_lat, obs_lon = sky2ll(el, az, heights[layer], position)
        res = hp.pixelfunc.get_interp_val(map_, obs_lon, obs_lat, lonlat=True)
        return res
    else:
        raise ValueError(
            f"The layer value must be integer and be in range [0, {nlayers - 1}]"
        )


def iri_star(pars):
    return iricore.IRI(*pars)


def calc_interp_val(data1, data2, dt1, dt2, dt):
    if dt1 == dt2:
        return data1

    x = np.asarray([0, (dt2 - dt1).total_seconds()])
    y = np.asarray([data1, data2])
    linmod = interp1d(x, y, axis=0)
    x_in = (dt - dt1).total_seconds()
    return linmod(x_in)


def calc_interp_val_star(pars):
    return calc_interp_val(*pars)
