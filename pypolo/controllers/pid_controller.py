from typing import Callable

import numpy as np

from .base_controller import BaseController


class PIDController(BaseController):
    r"""A Proportional–Integral–Derivative controller."""

    def __init__(
        self,
        error_fn: Callable,
        dt: float,
        kp: float,
        ki: float,
        kd: float,
    ):
        r"""Initialize the PID controller.

        Args:
            error_fn (Callable): The error function. It should take the current
                state and the goal state as arguments and return the error of
                shape (num_actions, ).
            dt (float): The time step.
            kp (float): The proportional gain.
            ki (float): The integral gain.
            kd (float): The derivative gain.

        """
        super().__init__()
        self.error_fn = error_fn
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0
        assert self.dt > 0.0, "The time step must be positive."

    def control(self, state: np.ndarray) -> np.ndarray:
        r"""Compute the control action using the PID algorithm.

        Args:
            state (np.ndarray): The current state of the system.

        Returns:
            float: The control action.

        """
        goal = self.goals[0]
        error = self.error_fn(state, goal)
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        action = (self.kp * error + self.ki * self.integral +
                  self.kd * derivative)
        return action
