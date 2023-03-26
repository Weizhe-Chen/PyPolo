from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from ..dynamics import IDynamics


class IRobot(metaclass=ABCMeta):
    """Interface of robots."""
    def __init__(
        self,
        init_state: np.ndarray,
        dynamics: IDynamics,
        tolerance: float,
        sampling_rate: float,
    ) -> None:
        """

        Parameters
        ----------
        init_state: np.ndarray, shape=(dim_states, ), dtype=np.float64
            Initial robot state.
        dynamics: IDynamics
            Robot dynamics.
        tolerance: float
            Tolerance of error from the current state to the goal state.
        sampling_rate : float
            Sampling rate.

        """
        self.state = init_state
        self.tolerance = tolerance
        self.sampling_locations = []
        self.goal_states = []
        self.dynamics = dynamics
        self.__cumulative_time = 0.0
        if sampling_rate <= 0.0:
            raise ValueError("Sampling rate must be positive.")
        self.sampling_dt = 1.0 / sampling_rate

    @property
    def has_goal(self) -> bool:
        """Has a goal state or not?

         Returns
         -------
         bool
            If True, at least has one goal state.

        """
        return len(self.goal_states) > 0

    def update(self, dist: float, action: np.ndarray) -> None:
        """Update state, observation, and goal states.

        Parameters
        ----------
        dist: float
            Distance to the first goal state.
        action: np.ndarray, shape=(dim_action,)
            The action applied to the current state.

        """
        # Update state
        self.state = self.dynamics.step(self.state, action)
        # Get sensing observation at a fixed rate.
        self.__cumulative_time += self.dynamics.dt
        if self.__cumulative_time > self.sampling_dt:
            self.sampling_locations.append([
                self.state[0],
                self.state[1],
            ])
            self.__cumulative_time = 0.0
        # Delete the first goal if we already achieved it.
        if self.has_goal and (dist < self.tolerance):
            self.goal_states = self.goal_states[1:]

    @abstractmethod
    def control(self) -> Tuple[float, np.ndarray]:
        """Compute control output, i.e. action.

        Returns
        -------
        dist: float
            Distance to the first goal state.
        action: np.ndarray
            Control output.

        """
        raise NotImplementedError

    def commit_data(self) -> np.ndarray:
        """Returns the sampling locations."""
        x_new = np.vstack(self.sampling_locations)
        self.sampling_locations = []
        return x_new
