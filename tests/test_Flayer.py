import unittest
import numpy as np

from test_config import DT, POSITION, ref_coords

from dionpy import FLayer


class TestFLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFLayer, self).__init__(*args, **kwargs)
        self.target = FLayer(DT, POSITION, hbot=150, htop=500,
                             nlayers=100, nside=32, pbar=False,
                             iriversion=20)
        self.el, self.az, self.elm, self.azm = ref_coords()

    def test_refr(self):
        refr_data = np.load("test_data/FLayer_refr.npy")
        calc_refr = self.target.refr(self.elm, self.azm, freq=40, troposphere=True)
        self.assertTrue(np.isclose(calc_refr, refr_data).all())