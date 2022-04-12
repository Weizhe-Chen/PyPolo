from abc import ABCMeta, abstractmethod

import numpy as np


class IDynamics(metaclass=ABCMeta):
    """Interface of dynamics models."""
    @abstractmethod
    def __init__(self, rate: float) -> None:
        """

        Parameters
        ----------
        rate : float
            Dynamics update rate.

        """
        self.rate = rate
        self.dt = 1.0 / rate

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Update the given `state` with `action`.

        Parameters
        ----------
        state : numpy.ndarray, shape=(num_states,)
            The current state.
        action: numpy.ndarray, shape=(num_actions,)
            The action applied to the current state.

        Returns
        -------
        numpy.ndarray, shape=(num_states,)
            The resulting state after applying `action` to `state`.

        """
        raise NotImplementedError
