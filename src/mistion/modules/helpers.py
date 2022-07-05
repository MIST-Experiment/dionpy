import numpy as np

class OrderError(Exception):
    """
    Exception indicating incorrect order of simulation routines.
    """

    pass


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


def check_latlon(lat, lon):
    if not -90 <= lat <= 90:
        raise ValueError("Latitude of the instrument must be in range [-90, 90]")
    if not -180 <= lon < 180:
        raise ValueError("Longitude of the instrument must be in range [-180, 180]")


def none_or_array(vals):
    if vals is None:
        return None
    return np.array(vals)


def generate_grid(el_start, el_end, az_start, az_end, gridsize):
    """
    Generates a grid of coordinates at which all the parameters will be calculated.

    Parameters
    ----------
    el_start : float
        The starting value of the sequence of elevations (in degrees).
    el_end : float
        The end value of the sequence of elevations (in degrees).
    az_start : float
        The starting value of the sequence of azimuths (in degrees).
    az_end : float
        The end value of the sequence of azimuths (in degrees).
    gridsize : int
        Resolution of the coordinate grid. The total number of points will be [gridsize x gridsize].
    """

    az_vals = np.linspace(az_start, az_end, gridsize, endpoint=True)
    el_vals = np.linspace(el_start, el_end, gridsize)
    # Alternative to az, el = np.meshgrid(az, el), then flatten
    az = np.repeat(az_vals, gridsize)
    el = np.tile(el_vals, gridsize)
    return el, az


def generate_plot_grid(el_start, el_end, az_start, az_end, gridsize):
    az_vals = np.linspace(az_start, az_end, gridsize, endpoint=True)
    el_vals = np.linspace(el_start, el_end, gridsize)
    el_rows, az_rows = np.meshgrid(el_vals, az_vals)
    return az_vals, az_rows, el_vals, el_rows