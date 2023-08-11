import unittest
import numpy as np

from test_config import DT, POSITION, ref_coords

from dionpy import DLayer


class TestDLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDLayer, self).__init__(*args, **kwargs)
        self.target = DLayer(DT, POSITION, hbot=60, htop=90,
                             nlayers=100, nside=32, pbar=False,
                             iriversion=20)
        self.el, self.az, self.elm, self.azm = ref_coords()

    def test_atten(self):
        atten_data = np.load("test_data/DLayer_atten.npy")
        calc_atten = self.target.atten(self.elm, self.azm, freq=40,
                                       col_freq='default', emission=False,
                                       troposphere=True)
        self.assertTrue(np.isclose(calc_atten, atten_data).all())

    def test_emiss(self):
        emiss_data = np.load("test_data/DLayer_emiss.npy")
        _, calc_emiss = self.target.atten(self.elm, self.azm, freq=40,
                                          col_freq='default', emission=True,
                                          troposphere=True)
        self.assertTrue(np.isclose(calc_emiss, emiss_data).all())