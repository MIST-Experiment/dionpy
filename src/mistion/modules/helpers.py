import numpy as np
from .ion_tools import srange
from pymap3d import aer2geodetic
import healpy as hp


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
