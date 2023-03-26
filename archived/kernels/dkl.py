import torch

from .nn import TwoHiddenLayerTanhNN
from .rbf import RBF
from .scaler import Scaler


class DKL(RBF):
    def __init__(
        self,
        amplitude: float,
        lengthscale: float,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
    ) -> None:
        """

        Parameters
        ----------
        amplitude : float
            Amplitude of RBF.
        lengthscale: float
            Lengthscale of RBF.
        dim_input: int
            Input dimension of the neural network.
        dim_hidden: int
            Hidden dimension of the neural network.
        dim_output: int
            Output dimension of the neural network.

        """
        super().__init__(amplitude, lengthscale)
        self.nn = TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
            softmax=False,
        ).double()
        self.scaler = Scaler(-1.0, 1.0)

    def input_warping(self, x):
        features = self.nn(x)
        scaled_features = self.scaler(features)
        return scaled_features

    def forward(
        self,
        x_1,
        x_2,
    ):
        features_1 = self.input_warping(x_1)
        features_2 = self.input_warping(x_2)
        scale = self.lengthscale
        dist = torch.cdist(features_1.div(scale), features_2.div(scale), p=2)
        cov_mat = self.amplitude * torch.exp(-0.5 * torch.square(dist))
        return cov_mat
