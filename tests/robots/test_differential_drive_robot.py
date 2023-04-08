import os
import math

import numpy as np
import pytest
import pyvista as pv

from pypolo.robots import DifferentialDriveRobot


@pytest.fixture(scope="module")
def robot() -> DifferentialDriveRobot:
    return DifferentialDriveRobot(hertz=60.0, state=np.zeros(5))


def test_error(robot: DifferentialDriveRobot) -> None:
    error = robot.error(x=1.0, y=1.0)
    expected = np.array([math.sqrt(2), np.pi / 4])
    assert np.allclose(error, expected)


def test_step(robot: DifferentialDriveRobot) -> None:
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    action = np.array([0.0, 0.0])
    new_state = robot.step(state=state, action=action)
    expected = np.array([1.0 / 60.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(new_state, expected)

    state = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    action = np.array([0.0, 0.0])
    new_state = robot.step(state=state, action=action)
    expected = np.array([1.0 / 60.0, 0.0, 1.0 / 60.0, 1.0, 1.0])
    assert np.allclose(new_state, expected)

    state = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    action = np.array([-0.01, -0.01])
    new_state = robot.step(state=state, action=action)
    expected = np.array([1.0 / 60.0, 0.0, 1.0 / 60.0, 0.99, 0.99])
    assert np.allclose(new_state, expected)


def test_visualization(robot: DifferentialDriveRobot, render: bool) -> None:
    if not render:
        return
    np.random.seed(1)
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_text("Differential Drive Robot", font_size=24)
    plane = pv.Plane(i_size=20, j_size=20)
    plotter.add_mesh(plane, show_edges=True, color="lightblue")

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
    save_path = os.path.join(os.path.dirname(__file__),
                             "test_differential_drive_robot.mp4")
    plotter.open_movie(save_path)

    # Animate the robot movement
    num_steps = 100
    for i in range(num_steps):
        action = np.random.randn(2)
        robot.take(action)
        rotated_box = box.rotate_z(robot.state[2] * 180 / math.pi)
        translated_box = rotated_box.translate(
            (robot.state[0], robot.state[1], 0))
        plotter.add_mesh(translated_box, color="red", opacity=i / num_steps)
        plotter.write_frame()
    plotter.close()
