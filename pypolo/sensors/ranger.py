import numpy as np
from typing import List

from ..utilities import GridMap
from .sensor import ISensor


class Ranger(ISensor):
    """Ranging sensor."""

    def __init__(
        self,
        rate: float,
        env: np.ndarray,
        env_extent: List[float],
        noise_scale: float,
    ) -> None:
        """

        Parameters
        ----------
        rate : float
            Sensing data update rate.
        env: np.ndarray, shape=(num_rows, num_cols)
            A matrix indicating the ground-truth values at different locations.
        env_extent: List[float], (xmin, xmax, ymin, ymax)
            Environment extent
        noise_scale: float
            Standard deviation of the observational Gaussian white noise.

        """
        super().__init__(rate)
        self.env = GridMap(env, env_extent)
        self.noise_scale = noise_scale

    def sense(
        self,
        states: np.ndarray,
        rng=None,
    ) -> np.ndarray:
        """Get sensor observations.

        Parameters
        ----------
        states : np.ndarray, shape=(num_samples, dim_state)
            Get sensor observatinos at the given states.
        rng : np.random.RandomState, optional
            Random number generator for making the observation noisy.

        Returns
        -------
        observations: np.ndarray, shape=(num_samples, )
            Observation at the given state.

        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        observations = self.env.get(states[:, 0], states[:, 1])
        if rng is not None:
            observations = rng.normal(loc=observations, scale=self.noise_scale)
        return observations
