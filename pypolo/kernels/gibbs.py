import torch

from ..utilities.linalg import softplus
from .kernel import IKernel
from .nn import TwoHiddenLayerTanhNN


def rbf(dist, lengthscale):
    """Radial Basis Function."""
    return torch.exp(-0.5 * torch.square(dist / lengthscale))


class Gibbs(IKernel):
    def __init__(
        self,
        amplitude: float,
        dim_input: int,
        dim_hidden: int,
    ) -> None:
        """

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.
        dim_input: int
            Input dimension of the neural network.
        dim_hidden: int
            Hidden dimension of the neural network.

        """
        super().__init__(amplitude)
        self.nn = TwoHiddenLayerTanhNN(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=1,
            softmax=False,
        ).double()

    def get_lengthscales(self, x):
        return softplus(self.nn(x))

    def forward(
        self,
        x_1,
        x_2,
    ):
        input_dim = x_1.size(1)
        lengthscales1 = self.get_lengthscales(x_1)
        lengthscales2 = self.get_lengthscales(x_2)
        cross_product = lengthscales1 * lengthscales2.t()
        cross_sum = lengthscales1.square() + lengthscales2.square().t()
        normalizer = (2.0 * cross_product / cross_sum)**(input_dim / 2.0)
        dist = torch.cdist(x_1, x_2, p=2)
        exponent = torch.exp(-input_dim * dist.square() / cross_sum)
        cov_mat = self.amplitude * normalizer * exponent
        return cov_mat
