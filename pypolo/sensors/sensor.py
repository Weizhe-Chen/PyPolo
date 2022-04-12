from abc import ABCMeta, abstractmethod

import numpy as np


class ISensor(metaclass=ABCMeta):
    """Sensor interface."""
    @abstractmethod
    def __init__(self, rate: float) -> None:
        """

        Parameters
        ----------
        rate : float
            Sensing data update rate.

        """
        if rate <= 0.0:
            raise ValueError("rate must be positive.")
        self.__rate = rate
        self.dt = 1.0 / rate

    @abstractmethod
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
        observations: np.ndarray, shape=(num_samples, dim_observation)
            Observation at the given state.

        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        # Return the noiseless observations at the given states and add some
        # observational noise if rng is not None.
        raise NotImplementedError
