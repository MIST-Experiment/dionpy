import unittest
import numpy as np

from test_config import DT, POSITION, ref_coords

from dionpy import IonLayer


class TestIonLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestIonLayer, self).__init__(*args, **kwargs)
        self.target = IonLayer(DT, POSITION, hbot=60, htop=500,
                               nlayers=100, nside=32, rdeg=20,
                               pbar=False, iriversion=20)
        self.el, self.az, self.elm, self.azm = ref_coords()

    def test_edens(self):
        edens_data = np.load("test_data/IonLayer_edens.npy")
        self.assertTrue(np.isclose(self.target.edens, edens_data, equal_nan=True).all())

    def test_etemp(self):
        etemp_data = np.load("test_data/IonLayer_etemp.npy")
        self.assertTrue(np.isclose(self.target.etemp, etemp_data, equal_nan=True).all())

    def test_ed_calc(self):
        ed_calc_data = np.load("test_data/IonLayer_ed_calc.npy")
        self.assertTrue(np.isclose(self.target.ed(self.elm, self.azm, layer=None),
                                   ed_calc_data, equal_nan=True).all())

    def test_et_calc(self):
        et_calc_data = np.load("test_data/IonLayer_et_calc.npy")
        self.assertTrue(np.isclose(self.target.et(self.elm, self.azm, layer=None),
                                   et_calc_data, equal_nan=True).all())
