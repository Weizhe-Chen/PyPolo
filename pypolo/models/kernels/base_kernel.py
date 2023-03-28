from abc import ABCMeta, abstractmethod

import torch
from torch.nn.parameter import Parameter

from pypolo.utils import torch_utils


class BaseKernel(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, amplitude: float, device_name: str) -> None:
        r"""Initializes a kernel with the specified amplitude and device name.

        Args:
            amplitude (float): The positive amplitude parameter of the kernel.
            device_name (str): The name of the PyTorch device to be used for
                computations.

        ??? note "Parameterization"

            An inverse softplus function transforms `amplitude` to
            `_free_amplitude` which is the parameter being optimized under
            the hood.

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
        r"""The amplitude of the kernel.

        Returns:
            torch.Tensor: The amplitude of the kernel.

        ??? note "Positivity"

            The positivity of amplitude is ensured by a softplus function.

        """
        return torch_utils.softplus(self._free_amplitude)

    @amplitude.setter
    def amplitude(self, amplitude: torch.Tensor) -> None:
        r"""The amplitude of the kernel.

        Args:
            amplitude (torch.Tensor): The new value of the amplitude.

        """
        with torch.inference_mode():
            self._free_amplitude.copy_(torch_utils.inv_softplus(amplitude))

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the diagonal elements of the self-covariance matrix.

        Args:
            x (torch.Tensor): Input data of shape (num_inputs, dim_inputs).

        Returns:
            torch.Tensor: Variance vector of shape (num_samples, 1).

        ??? tip "Why not extract the diagonal elements from the full matrix?"

            This method is more efficient than computing the full covariance
            matrix and indexing the diagonal elements.

        """
        return self.amplitude * torch.ones(
            x.size(0),
            1,
            dtype=self.dtype,
            device=self.device,
        )

    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the covariance matrix between two sets of inputs.

        Parameters:
            x1 (torch.Tensor): First input tensor of shape
                (num_inputs_1, dim_inputs_1).
            x2 (torch.Tensor): Second input tensor of shape
                (num_inputs_2, dim_inputs_2).

        Returns:
            cov_mat (torch.Tensor): Full covariance matrix of shape
                (num_inputs_1, num_inputs_2).

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by a subclass.

        """
        raise NotImplementedError
