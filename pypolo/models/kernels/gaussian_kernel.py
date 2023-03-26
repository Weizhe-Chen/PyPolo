import torch
from torch.nn.parameter import Parameter

from pypolo.utils import torch_utils
from .base_kernel import BaseKernel


class GaussianKernel(BaseKernel):

    def __init__(self, amplitude: float, lengthscale: float,
                 dtype: torch.dtype, device: torch.device) -> None:
        """

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.
        lengthscale: float
            Lengthscale hyper-parameter controlling "the size of the
            neighborhood" of one data point.

            A small `lengthscale` means that the value of a data point is
            similar to that of nearby points within a small radius.
            Larger `lengthscale` allows such similarity to expand across a
            larger radius.

        """
        super().__init__(amplitude, dtype, device)
        self.__free_lengthscale = Parameter(
            torch_utils.unconstraint(lengthscale, self.dtype, self.device))

    @property
    def lengthscale(self):
        """Lengthscale hyper-parameter."""
        return torch_utils.constraint(self.__free_lengthscale)

    @lengthscale.setter
    def lengthscale(self, lengthscale: float) -> None:
        with torch.no_grad():
            self.__free_lengthscale.copy_(
                torch_utils.unconstraint(lengthscale, self.dtype, self.device))

    def forward(self, x_1, x_2):
        scale = self.lengthscale
        dist = torch.cdist(x_1.div(scale), x_2.div(scale), p=2)
        cov_mat = self.amplitude * torch.exp(-0.5 * torch.square(dist))
        return cov_mat
