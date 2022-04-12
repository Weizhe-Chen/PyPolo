from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from ..utilities import linalg
from ..utilities import StandardScaler, MinMaxScaler


class IModel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of a probabilistic model."""

    @abstractmethod
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        noise: float,
        is_normalized: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        x_train: np.ndarray, shape=(num_samples, num_dims)
            Training inputs.
        y_train: np.ndarray, shape=(num_samples, 1)
            Training outputs.
        noise: float
            Observational noise variance.
        is_normalized: bool
            If True, inputs are normalized to (-1, 1) and outputs have zero
            mean and standard deviation.

        """
        super().__init__()
        self.__free_noise = Parameter(linalg.unconstraint(noise))
        self.is_normalized = is_normalized
        if self.is_normalized:
            self.set_scalers(x_train, y_train)
        self.set_data(x_train, y_train)

    def set_scalers(self, x_init: np.ndarray, y_init: np.ndarray) -> None:
        """Set input scaler `x_scaler` and output scaler `y_scaler`.

        Parameters
        ----------
        x_init: np.ndarray, shape=(num_samples, num_dims)
            Training inputs.
        y_init: np.ndarray, shape=(num_samples, 1)
            Training outputs.

        Attributes
        ----------
        x_scaler: MinMaxScaler
            Input scaler.
        y_scaler: StandardScaler
            Output scaler.

        """
        self.x_scaler = MinMaxScaler(x_init, expected_range=(-1.0, 1.0))
        self.y_scaler = StandardScaler(values=y_init)

    def set_data(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Set attributes `_x_train` and `_y_train`.

        Parameters
        ----------
        x_train: np.ndarray, shape=(num_samples, num_dims)
            Training inputs.
        y_train: np.ndarray, shape=(num_samples, 1)
            Training outputs.

        """
        self.check_shape(x_train, y_train)
        if self.is_normalized:
            x_train = self.x_scaler.preprocess(x_train)
            y_train = self.y_scaler.preprocess(y_train)
        self._x_train = torch.tensor(x_train, dtype=torch.float64)
        self._y_train = torch.tensor(y_train, dtype=torch.float64)

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        """Append new data to `x_train` and `y_train`.

        Parameters
        ----------
        x_new: np.ndarray, shape=(num_samples, num_dims)
            New training inputs.
        y_new: np.ndarray, shape=(num_samples, 1)
            New training outputs.

        """
        self.check_shape(x_new, y_new)
        if self.is_normalized:
            x_new = self.x_scaler.preprocess(x_new)
            y_new = self.y_scaler.preprocess(y_new)
        _x_new = torch.tensor(x_new, dtype=torch.float64)
        _y_new = torch.tensor(y_new, dtype=torch.float64)
        self._x_train = torch.vstack((self._x_train, _x_new))
        self._y_train = torch.vstack((self._y_train, _y_new))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the original`x_train` and `y_train`.
        """
        x_train = self._x_train.numpy()
        y_train = self._y_train.numpy()
        if self.is_normalized:
            x_train = self.x_scaler.postprocess(x_train)
            y_train = self.y_scaler.postprocess_mean(y_train)
        return x_train, y_train

    @staticmethod
    def check_shape(x_train, y_train):
        if x_train.ndim != 2:
            raise ValueError("x_train should be 2-dimensional.")
        if y_train.ndim != 2:
            raise ValueError("y_train should be 2-dimensional.")
        if y_train.shape[1] != 1:
            raise ValueError("Only support univariate output for now.")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train should have same length.")

    @property
    def num_train(self) -> int:
        """Number of training data."""
        return self._x_train.size(0)

    @abstractmethod
    def optimize(self, num_iter: int, verbose: bool = True) -> None:
        """Optimize all the parameters by minimizing the loss.

        Parameters
        ----------
        num_iter: int
            Number of optimization/training iterations.
        verbose: bool
            Print the optimization information or not?

        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x_test: np.ndarray,
        noise_free: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Make prediction.

        Parameters
        ----------
        x_test: np.ndarray, shape=(num_samples, num_dims)
            Test inputs.
        noise_free: bool, optional
            If True, predict the latent function values \(\mathbf{f}\).
            Otherwise, predict the noisy targets \(\mathbf{y}\).

        Returns
        -------
        mean: np.ndarray, shape=(num_samples, 1)
            Predictive mean.
        std: np.ndarray, shape=(num_samples, 1)
            Predictive standard deviation.

        """
        # Notes: don't forget to preprocess x_test and post-process mean
        # std.
        raise NotImplementedError

    @property
    def noise(self):
        """Observational noise variance."""
        return linalg.constraint(self.__free_noise)

    @noise.setter
    def noise(self, noise: float) -> None:
        """Set observational noise variance.

        Parameters
        ----------
        noise: float
            New observational noise variance.

        """
        with torch.no_grad():
            self.__free_noise.copy_(linalg.unconstraint(noise))
