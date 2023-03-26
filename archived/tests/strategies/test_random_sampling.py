import numpy as np

from pypolo.strategies import RandomSampling


def test_bezier():
    extent = [-10.0, 10.0, -10.0, 10.0]
    rng = np.random.RandomState(0)
    random_sampling = RandomSampling(task_extent=extent, rng=rng)
    actual = random_sampling.get(num_states=10000)
    np.testing.assert_allclose(
        actual.min(axis=0),
        np.array([-10, -10], dtype=np.float64),
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        actual.max(axis=0),
        np.array([10, 10], dtype=np.float64),
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        actual.mean(axis=0),
        np.array([0.0, 0.0], dtype=np.float64),
        atol=1e-1,
        rtol=1e-1,
    )

    #  # Uncomment the following code for visualization
    #  from matplotlib import pyplot as plt
    #  _, ax = plt.subplots()
    #  ax.set_xlim(extent[:2])
    #  ax.set_ylim(extent[2:])
    #  ax.scatter(
    #      actual[:, 0],
    #      actual[:, 1],
    #      color='tab:red',
    #      marker='o',
    #      label='Waypoints',
    #  )
    #  plt.tight_layout()
    #  plt.show()
