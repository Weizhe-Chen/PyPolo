import numpy as np

from pypolo.strategies import Bezier


def test_bezier():
    extent = [0.0, 50.0, 0.0, 50.0]
    rng = np.random.RandomState(0)
    bezier = Bezier(task_extent=extent, rng=rng)
    actual = bezier.get(num_states=10)
    expected = np.load("./tests/data/bezier_waypoints.npy")
    np.testing.assert_allclose(actual, expected)

    #  # Uncomment the following code for visualization
    #  from matplotlib import pyplot as plt
    #  ax = bezier.curve.plot(num_pts=100)
    #  ax.scatter(
    #      actual[:, 0],
    #      actual[:, 1],
    #      color='tab:red',
    #      marker='o',
    #      label='Waypoints',
    #  )
    #  control_points = bezier.generate_control_points()
    #  ax.scatter(
    #      control_points[0, :],
    #      control_points[1, :],
    #      color='tab:green',
    #      marker='*',
    #      label='Control Points',
    #  )
    #  for i, point in enumerate(control_points.T):
    #      ax.annotate(f"{i}", point + 0.5)
    #  ax.grid('on')
    #  plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=2)
    #  plt.tight_layout()
    #  plt.show()
