import os
import math

import numpy as np
import pytest
import pyvista as pv
from pyvista import examples
from tqdm import tqdm

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


def test_differential_drive_visualization(
    robot: DifferentialDriveRobot,
    render: bool,
) -> None:
    if not render:
        return
    np.random.seed(1)
    plotter = pv.Plotter(off_screen=True)
    plane = pv.Plane(i_size=20, j_size=20)
    robot_mesh = pv.read("./auv.stl")
    save_path = os.path.join(
        os.path.dirname(__file__),
        "pypolo_differential_drive_robot.gif",
    )
    plotter.open_gif(save_path)
    # Animate the robot movement
    num_steps = 30
    for i in tqdm(range(num_steps)):
        action = np.random.randn(2)
        robot.take(action)
        rotated_mesh = robot_mesh.rotate_z(robot.state[2] * 180 / math.pi)
        translated_mesh = rotated_mesh.translate(
            (robot.state[0], robot.state[1], 0))

        plotter.add_axes()
        plotter.add_text(f"Timestep: {i:03d}", font_size=18)
        plotter.add_mesh(plane, show_edges=True, color="white")
        plotter.add_mesh(translated_mesh, color="orange", smooth_shading=True)
        plotter.write_frame()
        plotter.clear()
        plotter.enable_lightkit()
    plotter.close()
