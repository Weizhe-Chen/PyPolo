import numpy as np
import pytest

from pypolo.robots import USV


def test_usv_init(usv):
    np.testing.assert_allclose(
        np.array([0, 0, 0], dtype=np.float64),
        usv.state,
    )
    np.testing.assert_almost_equal(usv.tolerance, 0.1)
    assert len(usv.sampling_locations) == 0
    assert len(usv.goal_states) == 0
    np.testing.assert_almost_equal(usv.max_lin_vel, 1.0)


def test_usv_init_with_exception():
    init_state = np.array([0, 0, 0], dtype=np.float64)
    with pytest.raises(ValueError):
        USV(
            init_state=init_state.reshape(-1, 1),  # wrong shape
            control_rate=10.0,
            max_lin_vel=1.0,
            tolerance=0.3,
            sampling_rate=1.0,
        )
    with pytest.raises(ValueError):
        USV(
            init_state=init_state.astype(np.int64),  # wrong type
            control_rate=10.0,
            max_lin_vel=1.0,
            tolerance=0.3,
            sampling_rate=1.0,
        )
    with pytest.raises(ValueError):
        USV(
            init_state=init_state,
            control_rate=0.0,  # should be positive
            max_lin_vel=1.0,
            tolerance=0.3,
            sampling_rate=1.0)
    with pytest.raises(ValueError):
        USV(
            init_state=init_state,
            control_rate=10.0,
            max_lin_vel=0.0,  # should be positive
            tolerance=0.3,
            sampling_rate=1.0)
    with pytest.raises(ValueError):
        USV(
            init_state=init_state,
            control_rate=10.0,
            max_lin_vel=1.0,
            tolerance=0.0,  # should be positive
            sampling_rate=1.0)


def test_robot_has_goal(usv):
    assert not usv.has_goal
    usv.goal_states.append(np.array([1.0, 0.0], dtype=np.float64))
    assert usv.has_goal


def test_usv_control(usv):
    dist, action = usv.control()
    np.testing.assert_almost_equal(dist, 1.0)
    np.testing.assert_almost_equal(action[0], np.tanh(1.0))
    np.testing.assert_almost_equal(action[1], 0.0)
    usv.goal_states[0] = np.array([1.0, 1.0], dtype=np.float64)
    dist, action = usv.control()
    np.testing.assert_almost_equal(dist, np.hypot(1.0, 1.0))
    np.testing.assert_almost_equal(action[0], np.tanh(1.0))
    np.testing.assert_almost_equal(action[1], 2.0 * np.arctan2(1.0, 1.0))


def test_robot_update(usv):
    usv.goal_states.append(np.array([8.0, 1.0], dtype=np.float64))
    states = [usv.state.copy()]
    while usv.has_goal:
        usv.update(*usv.control())
        states.append(usv.state.copy())
    states = np.array(states)
    expected_states = np.load("./tests/data/expected_states.npy")
    np.testing.assert_allclose(
        states,
        expected_states,
    )
    #  # Uncomment the following code block for visualization
    #  rows, cols = np.ogrid[0:6, 0:7]
    #  env = rows + cols
    #  env = env.astype(np.float64)
    #  import matplotlib.pyplot as plt
    #  extent = [-10.0, 10.0, -10.0, 10.0]
    #  fig, ax = plt.subplots()
    #  im = ax.imshow(env, extent=extent)
    #  fig.colorbar(im)
    #  ax.set_xlim([-10.0, 10.0])
    #  ax.set_ylim([-10.0, 10.0])
    #  ax.quiver(
    #      states[:, 0],
    #      states[:, 1],
    #      np.cos(states[:, 2]),
    #      np.sin(states[:, 2]),
    #      alpha=0.2,
    #  )
    #  ax.scatter(
    #      sampling_locations[:, 0],
    #      sampling_locations[:, 1],
    #  )
    #  plt.show()
