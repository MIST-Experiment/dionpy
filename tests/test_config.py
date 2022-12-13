from datetime import datetime
import numpy as np

DT = datetime(2019, 2, 12, 6, 20, 0)
POSITION = (0, 0, 0)


def ref_coords():
    el = np.linspace(0, 90, 100)
    az = np.linspace(0, 360, 100)
    elm, azm = np.meshgrid(el, az)
    return el, az, elm, azm
