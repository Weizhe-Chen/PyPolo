import numpy as np
from .strategy import IStrategy
from typing import List
from ..models import IModel


class RandomSampling(IStrategy):
    """Random sampling."""
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
    ) -> None:
        """

        Parameters
        ----------
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Bounding box of the sampling task workspace.
        rng: np.random.RandomState
            Random number generator if `get` has random operations.

        """
        super().__init__(task_extent, rng)

    def get(self, model=None, num_states: int = 1) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        num_states: int
            Number of goal states.

        Returns
        -------
        goal_states: np.ndarray, shape=(num_states, dim_states)
            Sampling goal states.

        """
        xs = self.rng.uniform(
            low=self.task_extent[0],
            high=self.task_extent[1],
            size=num_states,
        )
        ys = self.rng.uniform(
            low=self.task_extent[2],
            high=self.task_extent[3],
            size=num_states,
        )
        goal_states = np.column_stack((xs, ys))
        return goal_states
