#  import math
#
#  import numpy as np
#
from pypolo.robots import DifferentialDriveRobot
import numpy as np

np.random.seed(1)
#
#
#  def test_step():
#      robot = DifferentialDriveRobot(hertz=10, state=np.array([0, 0, 0]))
#      new_state = robot.step(np.array([0, 0, 0]), np.array([1, 0]))
#      assert np.allclose(new_state, np.array([0.1, 0, 0]))
#
#
#  def test_error():
#      robot = DifferentialDriveRobot(hertz=10, state=np.array([0, 0, 0]))
#      error = robot.error(np.array([0, 0, 0]), np.array([1, 1, 0]))
#      assert np.allclose(error, np.array([math.sqrt(2), np.pi / 4]))

#  import pytest
#  import numpy as np
#  import pyvista as pv
#
#
#  @pytest.fixture
#  def robot():
#      # Set up the robot with initial state and control rate
#      init_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
#      rate = 10.0  # Hz
#      robot = DifferentialDriveRobot(rate, init_state)
#      return robot
#
#
#  def test_robot_movement(robot):
#      # Set up the simulation environment
#      plotter = pv.Plotter()
#      plane = pv.Plane(i_size=10, j_size=10)
#      plotter.add_mesh(plane, color="white")
#
#      # Set up initial state and goal
#      goal = np.array([5.0, 5.0, np.pi / 2])  # [x, y, theta]
#
#      # Move the robot to the goal state
#      t = 0.0
#      while True:
#          # Get the error and action
#          error = robot.error(robot.state, goal)
#          action = np.array([error[0], error[1] * 10])  # v, omega
#
#          # Update the state of the robot
#          robot.state = robot.step(robot.state, action)
#
#          # Update the position of the robot in the simulation
#          pose = robot.state
#          robot_pos = pv.Arrow(start=[pose[0], pose[1], 0.0],
#                               direction=[np.cos(pose[2]),
#                                          np.sin(pose[2]), 0.0],
#                               scale=0.5,
#                               tip_length=0.3)
#          plotter.add_mesh(robot_pos, color="red")
#
#          # Check if the robot has reached the goal
#          if np.allclose(robot.state, goal, atol=0.1):
#              break
#
#          # Update time and render the scene
#          t += robot.control_dt
#          plotter.update()
#          plotter.show(auto_close=False)
#
#      # Check that the final robot state is equal to the goal state
#      assert np.allclose(robot.state, goal, atol=0.1)

import pytest
import numpy as np
import math
import pyvista as pv


@pytest.fixture(scope="module")
def robot():
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    rate = 10.0  # Hz
    return DifferentialDriveRobot(rate, initial_state)


def test_visualization(robot):
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_text("Differential Drive Robot", font_size=24)

    # Create a circle for the robot to move on
    circle_resolution = 100
    circle_radius = 5.0
    theta = np.linspace(0, 2 * np.pi, circle_resolution, endpoint=False)
    x = circle_radius * np.cos(theta)
    y = circle_radius * np.sin(theta)
    z = np.zeros_like(theta)
    circle = pv.PolyData(np.column_stack([x, y, z]))
    plotter.add_mesh(circle, color="lightblue", opacity=0.7)

    #  Create a box for the robot body
    body_length = 0.5
    body_width = 0.3
    body_height = 0.2
    box = pv.Box(bounds=[
        -body_length / 2,
        body_length / 2,
        -body_width / 2,
        body_width / 2,
        -body_height / 2,
        body_height / 2,
    ])
    plotter.open_gif("differential_drive_robot.gif")

    # Animate the robot movement
    num_steps = 100
    for i in range(num_steps):
        action = 0.1 * np.random.randn(2)
        robot.take(action)
        rotated_box = box.rotate_z(robot.state[2] * 180 / math.pi)
        translated_box = rotated_box.translate(
            (robot.state[0], robot.state[1], 0))
        plotter.add_mesh(translated_box, color="red", opacity=i / num_steps)
        plotter.write_frame()
    plotter.close()
