from abc import ABC, abstractmethod

import numpy as np

from ..sensors import BaseSensor


class BaseRobot(ABC):

    def __init__(self, rate: float, state: np.ndarray):
        self.state = state
        self.rate = rate
        self.control_dt = 1.0 / rate
        self.sensing_dt = 0.0
        self.sensors = []

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
            goal (np.ndarray): Goal state of the robot.

        Returns:
            np.ndarray: An error array of shape (num_actions, ).
        """
        raise NotImplementedError

    def move(self, action: np.ndarray) -> None:
        r"""Move the robot.

        Args:
            action (np.ndarray): Action to be taken by the robot.

        """
        self.state = self.step(self.state, action)

    def add_sensor(self, sensor: BaseSensor) -> None:
        r"""Add a sensor to the robot.

        Args:
            sensor (BaseSensor): Sensor to be added to the robot.

        """
        self.sensors.append(sensor)
