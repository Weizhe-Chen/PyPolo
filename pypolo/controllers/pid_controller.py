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
        error_threshold: float,
    ):
        r"""Initialize the PID controller.

        Args:
            error_fn (Callable): The robot's error function. It should take
                the goal [x, y] as arguments and return the error of
                shape (num_actions, ).
            dt (float): The time step.
            kp (float): The proportional gain.
            ki (float): The integral gain.
            kd (float): The derivative gain.
            error_threshold (float): If the error is less than this threshold,
                the goal is considered reached.

        """
        super().__init__()
        self.error_fn = error_fn
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_threshold = error_threshold
        self.integral = 0.0
        self.previous_error = 0.0

    def control(self) -> np.ndarray:
        r"""Compute the control action using the PID algorithm.

        Returns:
            float: The control action.

        """
        goal = self.goals[0]
        error = self.error_fn(x=goal[0], y=goal[1])
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        action = (self.kp * error + self.ki * self.integral +
                  self.kd * derivative)
        if np.linalg.norm(error) < self.error_threshold:
            self.goals = self.goals[1:]
        return action
