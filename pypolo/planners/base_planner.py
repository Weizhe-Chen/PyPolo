from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np


class BasePlanner(ABC):

    def __init__(self, workspace: List[float], objective: Callable) -> None:
        r"""Interface of a planner.

        Args:
            workspace (List[float]): Bounding box [xmin, xmax, ymin, ymax]
                of the planner's workspace.
            objective (Callable): The objective function.

        """
        self.workspace = workspace
        self.objective = objective

    @abstractmethod
    def plan(self, state: np.ndarray) -> np.ndarray:
        """Plan informative waypoint(s) given the robot state.

        Args:
            state (np.ndarray): The robot state.

        Returns:
            waypoints (np.ndarray): The resulting informative waypoint(s) of
                shape (num_waypoints, 2).

        """
        raise NotImplementedError
