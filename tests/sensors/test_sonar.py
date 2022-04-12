from pypolo.sensors import Sonar
import numpy as np
import pytest


def test_sonar_init(sonar):
    np.testing.assert_almost_equal(sonar.dt, 1.0)


def test_sonar_init_with_exception():
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    with pytest.raises(ValueError):
        Sonar(rate=0.0, env=env, env_extent=extent, noise_scale=0.1)


def test_sense(sonar):
    state = np.array([-10.0, -10.0])  # x = 0.0, y = 0.0
    actual = sonar.sense(state)
    np.testing.assert_almost_equal(actual, 0.0)
    state = np.array([10.0, -10.0])  # x = 6.0, y = 0.0
    actual = sonar.sense(state)
    np.testing.assert_almost_equal(actual, 6.0)
    state = np.array([10.0, 10.0])  # x = 6.0, y = 5.0
    actual = sonar.sense(state)
    np.testing.assert_almost_equal(actual, 11.0)
    state = np.array([-10.0, 10.0])  # x = 0.0, y = 5.0
    actual = sonar.sense(state)
    np.testing.assert_almost_equal(actual, 5.0)


def test_sense_with_noise(sonar):
    rng = np.random.RandomState()
    # Lower left
    states = np.repeat(np.array([[-10.0, -10.0]]), repeats=100, axis=0)
    actuals = sonar.sense(states, rng)
    np.testing.assert_almost_equal(actuals.mean(), 0.0, decimal=1)
    np.testing.assert_almost_equal(actuals.std(), sonar.noise_scale, decimal=1)
    # Lower right
    states = np.repeat(np.array([[10.0, -10.0]]), repeats=100, axis=0)
    actuals = sonar.sense(states, rng)
    np.testing.assert_almost_equal(actuals.mean(), 6.0, decimal=1)
    np.testing.assert_almost_equal(actuals.std(), sonar.noise_scale, decimal=1)
    # Upper right
    states = np.repeat(np.array([[10.0, 10.0]]), repeats=100, axis=0)
    actuals = sonar.sense(states, rng)
    np.testing.assert_almost_equal(actuals.mean(), 11.0, decimal=1)
    np.testing.assert_almost_equal(actuals.std(), sonar.noise_scale, decimal=1)
    # Upper left
    states = np.repeat(np.array([[-10.0, 10.0]]), repeats=100, axis=0)
    actuals = sonar.sense(states, rng)
    np.testing.assert_almost_equal(actuals.mean(), 5.0, decimal=1)
    np.testing.assert_almost_equal(actuals.std(), sonar.noise_scale, decimal=1)
