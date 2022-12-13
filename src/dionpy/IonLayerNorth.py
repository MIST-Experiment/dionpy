from __future__ import annotations

import itertools
from multiprocessing import cpu_count, Pool
from typing import Union, List

import iricore
import numpy as np
from tqdm import tqdm

from .IonLayer import IonLayer
from .modules.parallel import echaim_star


class IonLayerNorth(IonLayer):
    """
    A model of a layer of specific height range in the ionosphere based on the ECHAIM model.
    """




