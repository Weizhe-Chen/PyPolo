from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parameter import Parameter

from pypolo.utils import torch_utils


class BaseKernel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of a kernel."""

    @abstractmethod
    def __init__(self, amplitude: float, device_name: str) -> None:
        """Initialize a kernel.

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.
        device_name : str
            PyTorch device name.
        """
        super().__init__()
        self.dtype, self.device = torch_utils.get_dtype_and_device(device_name)
        self._free_amplitude = Parameter(
            torch_utils.inv_softplus(
                torch.as_tensor(
                    amplitude,
                    dtype=self.dtype,
                    device=self.device,
                )))

    @property
    def amplitude(self) -> torch.Tensor:
        return torch_utils.softplus(self._free_amplitude)

    @amplitude.setter
    def amplitude(self, amplitude: torch.Tensor) -> None:
        with torch.inference_mode():
            self._free_amplitude.copy_(torch_utils.inv_softplus(amplitude))

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Diagonal elements of a self-covariance matrix, i.e., variance.

        Parameters
        ----------
        x : torch.Tensor shape=(num_inputs, dim_inputs]
            Inputs.

        Returns
        -------
        torch.Tensor shape=(num_samples, 1)
            Variance vector.

        Notes
        -----
        This method is more efficient than computing the covariance matrix
        and slicing the diagonal elements.

        """
        return self.amplitude * torch.ones(
            x.size(0),
            1,
            dtype=self.dtype,
            device=self.device,
        )

    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix.

        Parameters
        ----------
        x1 : torch.Tensor shape=(num_inputs_1, dim_inputs_1)
            The first input.
        x2 : torch.Tensor shape=(num_inputs_2, dim_inputs_2)
            The second input.

        Returns
        -------
        cov_mat: torch.Tensor shape=(num_inputs_1, num_inputs_2)
            Full covariance matrix.
        """
        raise NotImplementedError
