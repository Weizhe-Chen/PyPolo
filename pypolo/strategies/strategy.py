from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from ..models import IModel


class IStrategy(metaclass=ABCMeta):
    """Sampling strategy."""
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
        self.task_extent = task_extent
        self.rng = rng

    @abstractmethod
    def get(self, model=None, num_states: int = 1) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        model: IModel, optional
            A probabilistic model that provides `mean` and `std` via `forward`.
        num_states: int,
            Number of goal states.

        Returns
        -------
        goal_states: np.ndarray, shape=(num_states, dim_states)
            Sampling goal states.

        """
        raise NotImplementedError
