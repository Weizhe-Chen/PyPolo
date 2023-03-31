from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    r"""Interface of a controller."""

    def __init__(self):
        r"""Initialize the controller."""
        self.goals = np.array([])

    def set_goals(self, goals: np.ndarray) -> None:
        r"""Set the goals of the controller.

        Args:
            goals (np.ndarray): The goals of the controller.

        """
        self.goals = goals

    @property
    def has_goals(self) -> bool:
        r"""Check if the controller has goals.

        Returns:
            bool: True if the controller has goals, False otherwise.

        """
        return len(self.goals) > 0

    @abstractmethod
    def control(self, state: np.ndarray) -> np.ndarray:
        r"""Compute the control action.

        Args:
            state (np.ndarray): The current state of the system.

        Returns:
            np.ndarray: The control action.

        """
        raise NotImplementedError
