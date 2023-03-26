from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parameter import Parameter

from pypolo.utils import torch_utils


class BaseKernel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of a kernel."""

    @abstractmethod
    def __init__(self, amplitude: float, dtype: torch.dtype,
                 device: torch.device) -> None:
        """Initialize a kernel.

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self._free_amplitude = Parameter(
            torch_utils.unconstraint(amplitude, self.dtype, self.device))

    @property
    def amplitude(self):
        """Amplitude hyper-parameter for scaling the kernel values."""
        return torch_utils.constraint(self._free_amplitude)

    @amplitude.setter
    def amplitude(self, amplitude: float) -> None:
        with torch.no_grad():
            self._free_amplitude.copy_(
                torch_utils.unconstraint(amplitude, self.dtype, self.device))

    def diag(self, x):
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
        return self.amplitude * torch.ones(
            x.shape[0], 1, dtype=self.dtype, device=self.device)

    @abstractmethod
    def forward(self, x_1, x_2):
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
