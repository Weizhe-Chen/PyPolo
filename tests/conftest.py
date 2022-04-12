import numpy as np
import pytest

from pypolo.dynamics import DubinsCar
from pypolo.utilities import GridMap


@pytest.fixture(scope="module")
def dubins_car() -> DubinsCar:
    return DubinsCar(rate=10.0)


@pytest.fixture(scope="module")
def grid_map() -> GridMap:
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    return GridMap(matrix=env, extent=extent)
