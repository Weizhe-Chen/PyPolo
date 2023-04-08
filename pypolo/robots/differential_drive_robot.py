import numpy as np
import math

from .base_robot import BaseRobot


class DifferentialDriveRobot(BaseRobot):
    r"""A robot with differential drive wheels."""

    def __init__(self, hertz: float, state: np.ndarray):
        r"""Initializes the robot with a control rate and initial state.

        Args:
            hertz (float): Control rate of the robot [#cycles / second].
            state (np.ndarray): Initial state of the robot [x, y, θ, v, ω].
                Shape: (num_states, ).

        """
        assert hertz > 0.0
        assert state.shape == (5, )
        self.state = state
        self.hertz = hertz
        self.control_dt = 1.0 / hertz

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        r"""Transition function for the robot.

        Args:
            state (np.ndarray): Current state of the robot [x, y, θ, v, ω]].
            action (np.ndarray): Action to be taken by the robot [dv, dω].

        Returns:
            np.ndarray: New state of the robot.

        """
        assert state.shape == (5, )
        assert action.shape == (2, )
        x, y, theta, v, omega = state
        delta_v, delta_omega = action
        delta_x = v * math.cos(theta) * self.control_dt
        delta_y = v * math.sin(theta) * self.control_dt
        delta_theta = omega * self.control_dt
        new_x = x + delta_x
        new_y = y + delta_y
        new_theta = (theta + delta_theta) % (2 * math.pi)
        new_v = v + delta_v
        new_omega = omega + delta_omega
        return np.array([new_x, new_y, new_theta, new_v, new_omega])

    def error(self, x: float, y: float) -> np.ndarray:
        r"""Given a goal location [x, y], returns an error array with the same
        shape as the action space.

        Args:
            x (float): Goal x position.
            y (float): Goal y position.

        Returns:
            np.ndarray: An error array of shape (num_actions, ).

        """
        error_x = x - self.state[0]
        error_y = y - self.state[1]
        linear_error = math.sqrt(error_x**2 + error_y**2)
        error_omega = math.atan2(error_y, error_x) - self.state[2]
        angular_error = (error_omega + math.pi) % (2 * math.pi) - math.pi
        return np.array([linear_error, angular_error])
