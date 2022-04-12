import pytest

from pypolo.dynamics import DubinsCar


@pytest.fixture(scope="module")
def dubins_car() -> DubinsCar:
    return DubinsCar(rate=10.0)
