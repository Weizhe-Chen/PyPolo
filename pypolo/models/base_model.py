from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseModel(ABC):
    """Interface of a probabilistic model."""

    @abstractmethod
    def learn(self,
              x_new: np.ndarray,
              y_new: np.ndarray,
              num_iter: int,
              verbose: bool = True) -> None:
        """Optimize the model parameters.

        Parameters
        ----------
        x_new: np.ndarray, shape=(num_samples, dim_input)
            New training inputs.
        y_new: np.ndarray, shape=(num_samples, dim_output)
            New training outputs.
        num_iter: int
            Number of optimization/training iterations.
        verbose: bool
            Print the optimization information or not?

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Make prediction.

        Parameters
        ----------
        x_test: np.ndarray, shape=(num_samples, num_dims)
            Test inputs.

        Returns
        -------
        mean: np.ndarray, shape=(num_samples, 1)
            Predictive mean.
        std: np.ndarray, shape=(num_samples, 1)
            Predictive standard deviation.
        """
        raise NotImplementedError
