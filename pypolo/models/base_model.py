from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch


class BaseModel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of a probabilistic model."""

    @abstractmethod
    def __init__(self, dtype="float64",device="cpu") -> None:
        super().__init__()
        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError("dtype should be either float64 or float32.")
        self.device = device

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        """Add new input-output pairs to the model.

        Parameters
        ----------
        x_new: np.ndarray, shape=(num_samples, dim_input)
            New training inputs.
        y_new: np.ndarray, shape=(num_samples, dim_output)
            New training outputs.

        """
        raise NotImplementedError

    @staticmethod
    def _check_shape(x_train, y_train):
        if x_train.ndim != 2:
            raise ValueError("x_train should be 2-dimensional.")
        if y_train.ndim != 2:
            raise ValueError("y_train should be 2-dimensional.")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train should have same length.")
    @abstractmethod
    def learn(self, num_iter: int, verbose: bool = True) -> None:
        """Optimize the model parameters.

        Parameters
        ----------
        num_iter: int
            Number of optimization/training iterations.
        verbose: bool
            Print the optimization information or not?

        """
        raise NotImplementedError

    @abstractmethod
    def forward( self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
