import numpy as np
import pytest

from pypolo.sensors import EnvironmentalSensor
from pypolo.utils import TensorMap


@pytest.fixture
def sensor():
    return EnvironmentalSensor(rate=10, noise_scale=0.01)


def test_sense(sensor):
    robot_state = np.array([0.5, 0.5, 0.0])
    env_tensor = np.ones((100, 100, 1))
    env_state = TensorMap(env_tensor,
                          origin=np.array([0.0, 0.0, 0.0]),
                          resolution=0.1)
    obs = sensor.sense(robot_state, env_state)
    assert isinstance(obs, np.ndarray)
    assert obs == pytest.approx(1.0, rel=0.1)
