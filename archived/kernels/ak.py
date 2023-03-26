import numpy as np
import torch

from .kernel import IKernel
from .nn import TwoHiddenLayerTanhNN


def rbf(dist, lengthscale):
    """Radial Basis Function."""
    return torch.exp(-0.5 * torch.square(dist / lengthscale))


class AK(IKernel):

    def __init__(
        self,
        amplitude: float,
        lengthscales: np.ndarray,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
    ) -> None:
        """

        Parameters
        ----------
        amplitude : float
            Positive amplitude parameter of a kernel.
        lengthscales: np.ndarray, shape=(num_lengthscales, ), dtype=np.float64
            Primitive lengthscales.
        dim_input: int
            Input dimension of the neural network.
        dim_hidden: int
            Hidden dimension of the neural network.
        dim_output: int
            Output dimension of the neural network.

        """
        super().__init__(amplitude)
        self.lengthscales = torch.tensor(lengthscales)
        np.set_printoptions(precision=2)
        print("Primitive lengthscales: ", self.lengthscales.numpy())
        self.nn = TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
        ).double()

    @property
    def num_lengthscales(self):
        return len(self.lengthscales)

    def get_representations(self, x):
        z = self.nn(x)
        representations = z / z.norm(dim=1, keepdim=True)
        return representations

    def forward(
        self,
        x_1,
        x_2,
    ):
        dist = torch.cdist(x_1, x_2)
        repre1 = self.get_representations(x_1)
        repre2 = self.get_representations(x_2)
        cov_mat = 0.0
        for i in range(self.num_lengthscales):
            attention_lengthscales = torch.outer(repre1[:, i], repre2[:, i])
            cov_mat += rbf(dist, self.lengthscales[i]) * attention_lengthscales
        attention_inputs = repre1 @ repre2.t()
        cov_mat *= self.amplitude * attention_inputs
        return cov_mat
