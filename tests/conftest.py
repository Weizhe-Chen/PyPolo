import numpy as np
import pytest

from pypolo.dynamics import DubinsCar
from pypolo.utilities import GridMap
from pypolo.kernels import RBF


@pytest.fixture(scope="module")
def dubins_car() -> DubinsCar:
    return DubinsCar(rate=10.0)


@pytest.fixture(scope="module")
def grid_map() -> GridMap:
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    return GridMap(matrix=env, extent=extent)

@pytest.fixture(scope="module")
def rbf() -> RBF:
    return RBF(amplitude=1.0, lengthscale=1.0)
