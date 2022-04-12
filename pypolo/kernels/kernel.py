from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parameter import Parameter

from ..utilities import linalg


class IKernel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of kernels."""
    @abstractmethod
    def __init__(self, amplitude: float) -> None:
        """

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.

        Notes
        -----
        This class should not be initialized. It defines interface of kernels
        and implements some common functionalities.

        """
        super().__init__()
        self.__free_amplitude = Parameter(linalg.unconstraint(amplitude))

    @property
    def amplitude(self):
        """Amplitude hyper-parameter for scaling the kernel values."""
        return linalg.constraint(self.__free_amplitude)

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        with torch.no_grad():
            self.__free_amplitude.copy_(linalg.unconstraint(amplitude))

    def diag(
            self,
            x,
    ):
        """Diagonal elements of a self-covariance matrix, i.e., variance.

        Parameters
        ----------
        x : TensorType["num_samples", "num_dims"]
            Inputs.

        Returns
        -------
        TensorType["num_samples", 1]
            Variance vector.

        Notes
        -----
        This method is more efficient than computing the covariance matrix
        and then retrieve the diagonal elements.

        """
        return self.amplitude * torch.ones(x.size(0), 1, dtype=torch.float64)

    @abstractmethod
    def forward(
        self,
        x_1,
        x_2,
    ):
        """Compute covariance matrix.

        Parameters
        ----------
        x_1 : TensorType["num_samples_1", "num_dims"]
            The first input.
        x_2 : TensorType["num_samples_2", "num_dims"]
            The second input.

        Returns
        -------
        cov_mat: TensorType["num_samples_1", "num_samples_2"]
            Full covariance matrix.

        """
        raise NotImplementedError
