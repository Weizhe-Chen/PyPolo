from typing import Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

from pypolo.utils import torch_utils

from .base_kernel import BaseKernel


class GaussianKernel(BaseKernel):

    def __init__(
        self,
        lengthscale: Union[float, list, np.ndarray, torch.Tensor],
        amplitude: float,
        device_name: str,
    ) -> None:
        """
        Parameters
        ----------
        lengthscale: float
            Hyper-parameter lengthscale of a Gaussian kernel.
        amplitude : float
            Positive amplitude parameter of a kernel.
        device_name : str
            PyTorch device name.

        Notes
        -----
        A large `lengthscale` means that the output value of a data point is
        similar to that of nearby points.
        """
        super().__init__(amplitude, device_name)
        self._free_lengthscale = Parameter(
            torch_utils.inv_softplus(
                torch.as_tensor(
                    lengthscale,
                    dtype=self.dtype,
                    device=self.device,
                )))

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch_utils.softplus(self._free_lengthscale)

    @lengthscale.setter
    def lengthscale(self, lengthscale: torch.Tensor) -> None:
        with torch.inference_mode():
            self._free_lengthscale.copy_(torch_utils.inv_softplus(lengthscale))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = self.lengthscale
        dist = torch.cdist(x1.div(scale), x2.div(scale), p=2)
        cov_mat = self.amplitude * torch.exp(-0.5 * torch.square(dist))
        return cov_mat
