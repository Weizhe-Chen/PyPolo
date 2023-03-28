from typing import Union

import numpy as np
import torch
from pypolo.utils import torch_utils
from torch.nn.parameter import Parameter

from .base_kernel import BaseKernel


class GaussianKernel(BaseKernel):

    def __init__(
        self,
        lengthscale: Union[float, list, np.ndarray, torch.Tensor],
        amplitude: float,
        device_name: str,
    ) -> None:
        r"""Initialize a Gaussian kernel.

        Args:
            lengthscale (Union[float, list, np.ndarray, torch.Tensor]):
                Positive hyper-parameter lengthscale.
            amplitude (float): Positive hyper-parameter amplitude.
            device_name (str): PyTorch device name.

        ??? note "Intuitive understanding of lengthscale"

            The covariance function, a.k.a. kernel, expresses the correlation
            between the function values at two input points as a function of
            the distance between those points. The larger the lengthscale,
            the smoother the resulting function, since nearby points will have
            similar function values.

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
        r"""The lengthscale hyper-parameter.

        Returns:
            torch.Tensor: The lengthscale of the kernel.

        ??? note "Positivity"

            The positivity of lengthscale is ensured by a softplus function.

        """
        return torch_utils.softplus(self._free_lengthscale)

    @lengthscale.setter
    def lengthscale(self, lengthscale: torch.Tensor) -> None:
        r"""The lengthscale hyper-parameter.

        Args:
            lengthscale (torch.Tensor): The new value of the lengthscale.

        """
        with torch.inference_mode():
            self._free_lengthscale.copy_(torch_utils.inv_softplus(lengthscale))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = self.lengthscale
        dist = torch.cdist(x1.div(scale), x2.div(scale), p=2)
        cov_mat = self.amplitude * torch.exp(-0.5 * torch.square(dist))
        return cov_mat
