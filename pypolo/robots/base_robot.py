from abc import ABC, abstractmethod

import numpy as np


class BaseRobot(ABC):
    r"""Base class for a robot."""

    def __init__(self, rate: float, state: np.ndarray):
        r"""Initializes the robot with a control rate and initial state.

        Args:
            rate (float): Control rate of the robot.
            state (np.ndarray): Initial state of the robot.
                Shape: (num_states, ).

        """
        self.state = state
        self.rate = rate
        self.control_dt = 1.0 / rate
        self.sensing_dt = 0.0

    def take(self, action: np.ndarray) -> None:
        r"""Takes an action and updates the state of the robot.

        Args:
            action (np.ndarray): Action to be taken by the robot.

        """
        self.state = self.step(self.state, action)

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        r"""Transition function for the robot.

        Args:
            action (np.ndarray): Action to be taken by the robot.

        Returns:
            np.ndarray: New state of the robot.

        """
        raise NotImplementedError

    @abstractmethod
    def error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        r"""Error function for the robot.

        Args:
            state (np.ndarray): Current state of the robot.
                Shape: (num_states, ).
            goal (np.ndarray): Goal state of the robot.
                Shape: (num_states, ).

        Returns:
            np.ndarray: An error array of shape (num_actions, ).
        """
        raise NotImplementedError
