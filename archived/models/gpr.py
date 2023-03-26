from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..kernels import IKernel
from ..utilities import linalg
from .model import IModel


class GPR(IModel):
    """Gaussian Process Regression."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        kernel: IKernel,
        noise: float,
        lr_hyper: float = 0.01,
        lr_nn: float = 0.001,
        jitter: float = 1e-6,
        is_normalized: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        x_train: np.ndarray, shape=(num_samples, num_dims)
            Training inputs.
        y_train: np.ndarray, shape=(num_samples, 1)
            Training outputs.
        kernel: IKernel
            Covariance function.
        noise: float
            Observational noise variance.
        lr_hyper: float = 0.01
            Learning rate of hyper-parameters.
        lr_nn: float = 0.001
            Learning rate of neural network parameters in non-stationary
            kernels.
        jitter: float = 1e-6
            Small positive number added to the digonal elements of covariance
            matrix for preventing Cholesky decomposition failure.
            `jitter` is also the min predictive variance.
        is_normalized: bool
            If True, inputs and outputs are normalized.

        """
        super().__init__(x_train, y_train, noise, is_normalized)
        self.kernel = kernel
        self.initialize_optimizers(lr_hyper, lr_nn)
        self.jitter = jitter

    def initialize_optimizers(self, lr_hyper: float, lr_nn: float) -> None:
        """Initialize optimizers for hyper-parameters and, optinally,
        neural network parameters in nonstationary kernels.

        Parameters
        ----------
        lr_hyper: float = 0.01
            Learning rate of hyper-parameters.
        lr_nn: float = 0.001
            Learning rate of neural network parameters in non-stationary
            kernels.

        """
        hyper_params, nn_params = [], []
        for name, param in self.named_parameters():
            if "nn" in name:
                nn_params.append(param)
            else:
                hyper_params.append(param)
        self.opt_hyper = torch.optim.Adam(hyper_params, lr=lr_hyper)
        if nn_params:
            self.opt_nn = torch.optim.Adam(nn_params, lr=lr_nn)
        else:
            self.opt_nn = None

    def compute_common(self):
        r"""Compute and cache common terms for `compute_loss` and `predict`.

        Attributes
        ----------
        L: TensorType([num_samples, num_samples], torch.float64)
            Lower Cholesky factor of the training covariance matrix:
            \[
            \mathbf{L}
            =\mathtt{Cholesky}(\mathbf{K}_{y})
            =\mathtt{Cholesky}(\mathbf{K}+\sigma_{n}^{2}).
            \]
        iK_y: TensorType([num_samples, 1], torch.float64)
            Inverse training covariance matrix multiplied by training outputs:
            \[
            \mathbf{K}^{-1}\mathbf{y}.
            \]
        """
        K = self.kernel(self._x_train, self._x_train)
        K.diagonal().add_(self.noise)
        L = linalg.robust_cholesky(K, jitter=self.jitter)
        iK_y = torch.cholesky_solve(self._y_train, L, upper=False)
        return L, iK_y

    def compute_loss(self):
        r"""Compute training loss.

        Returns
        -------
        TensorType([], torch.float64)
            Negative log marginalized likelihood:
            \[-\log{p(\mathbf{y}|X)}=
            \frac{1}{2}\mathbf{y}^{\intercal}\mathbf{K}_{y}\mathbf{y}
            +\frac{1}{2}\log{|\mathbf{K}_{y}|}
            +\frac{n}{2}\log{2\pi}.\]

        """
        L, iK_y = self.compute_common()
        quadratic = torch.sum(self._y_train * iK_y)
        logdet = L.diag().square().log().sum()
        constant = self.num_train * np.log(2 * np.pi)
        return 0.5 * (quadratic + logdet + constant)

    def optimize(self,
                 num_iter: int,
                 verbose: bool = True,
                 writer=None) -> None:
        """Optimize all the parameters by minimizing the loss.

        Parameters
        ----------
        num_iter: int
            Number of optimization/training iterations.
        verbose: bool
            Print the optimization information or not?

        """
        self.train()
        pbar = range(num_iter)
        if verbose:
            pbar = tqdm(pbar)
        for i in pbar:
            self.opt_hyper.zero_grad()
            if self.opt_nn is not None:
                self.opt_nn.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.opt_hyper.step()
            if self.opt_nn is not None:
                self.opt_nn.step()
            if verbose:
                pbar.set_description(f"Iter: {i:02d} loss: {loss.item(): .2f}")
            if writer is not None:
                writer.add_scalar('loss', loss.item(), i)
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name, param.grad, i)
        self.eval()

    def forward(
        self,
        x_test: np.ndarray,
        noise_free: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Make prediction.

        Parameters
        ----------
        x_test: np.ndarray, shape=(num_samples, num_dims)
            Test inputs.
        noise_free: bool, optional
            If True, predict the latent function values \(\mathbf{f}\).
            Otherwise, predict the noisy targets \(\mathbf{y}\).

        Returns
        -------
        mean: np.ndarray, shape=(num_samples, 1)
            Predictive mean.
        std: np.ndarray, shape=(num_samples, 1)
            Predictive standard deviation.

        """
        # Pre-processing
        if self.is_normalized:
            x_test = self.x_scaler.preprocess(x_test)

        _x_test = torch.tensor(x_test, dtype=torch.float64)
        # Prediction
        with torch.no_grad():
            L, iK_y = self.compute_common()
            Ksn = self.kernel(_x_test, self._x_train)
            Kss_diag = self.kernel.diag(_x_test)
            iL_Kns = torch.linalg.solve_triangular(L, Ksn.t(), upper=False)
            _mean = Ksn @ iK_y
            var = Kss_diag - iL_Kns.square().sum(0).view(-1, 1)
            # TODO: variance might be zero when lengthscale is too large.
            if torch.any(var <= 0.0):
                print(var.ravel().numpy())
                raise ValueError("Predictive variance <= 0.0!")
            var.clamp_(min=self.jitter)
            if not noise_free:
                var += self.noise
            _std = var.sqrt()
        mean = _mean.numpy()
        std = _std.numpy()
        # Post-processing
        if self.is_normalized:
            mean = self.y_scaler.postprocess_mean(mean)
            std = self.y_scaler.postprocess_std(std)
        return mean, std
