import unittest
import numpy as np

from test_config import DT, POSITION

from dionpy import IonLayer


class TestIonLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestIonLayer, self).__init__(*args, **kwargs)
        self.target = IonLayer(DT, POSITION, hbot=90, htop=500,
                               nlayers=100, nside=60, rdeg=20,
                               pbar=False, iriversion=20)

    def test_edens(self):
        edens_data = np.load("test_data/IonLayer_edens.npy")
        self.assertTrue(np.isclose(self.target.edens, edens_data).all())

    def test_etemp(self):
        etemp_data = np.load("test_data/IonLayer_etemp.npy")
        self.assertTrue(np.isclose(self.target.etemp, etemp_data).all())
