from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..utils import torch_utils
from .base_model import BaseModel
from .kernels import BaseKernel


class GPRModel(BaseModel, nn.Module):

    def __init__(
        self,
        device_name,
        kernel: BaseKernel,
        noise: float,
        lr_hyper: float = 0.01,
        lr_nn: float = 0.001,
        jitter: float = 1e-6,
    ) -> None:
        r"""Gaussian Process Regression.

        Args:
            device_name (str): The name of the device to run the model.
            kernel (BaseKernel): The kernel function.
            noise (float): The noise variance of the Gaussian likelihood.
            lr_hyper (float, optional): Learning rate of hyper-parameters.
            lr_nn (float, optional): Learning rate of network parameters.
            jitter (float, optional): The jitter to add to the diagonal of the
                covariance matrix. Defaults to 1e-6.

        ??? note "What is jitter and why is it necessary?"

            Jitter is a small positive number added to the diagonal of the
            covariance matrix to ensure that it is positive definite.
            This is necessary because the covariance matrix is not always
            positive definite due to numerical errors.

        """
        BaseModel.__init__(self, device_name)
        nn.Module.__init__(self)
        self.kernel = kernel
        self._free_noise = Parameter(
            torch_utils.inv_softplus(
                torch.as_tensor(
                    noise,
                    dtype=self.dtype,
                    device=self.device,
                )))
        self._init_optimizers(lr_hyper, lr_nn)
        self.jitter = jitter

    def learn(self,
              x_new: np.ndarray,
              y_new: np.ndarray,
              num_iter: int,
              verbose: bool = True,
              writer: Union[SummaryWriter, None] = None) -> None:
        r"""Optimizes the model parameters.

        Args:
            x_new (np.ndarray): New training inputs of shape
                (num_inputs, dim_inputs).
            y_new (np.ndarray): New training outputs of shape
                (num_outputs, dim_outputs).
            num_iter (int): Number of optimization/training iterations.
            verbose (bool): Print the optimization information or not?
            writer (Union[SummaryWriter, None]): Tensorboard writer.

        """
        self._add_data(x_new, y_new)
        self.train()
        progress_bar = tqdm(range(num_iter), disable=not verbose)
        for i in progress_bar:
            self.opt_hyper.zero_grad()
            if self.opt_nn is not None:
                self.opt_nn.zero_grad()
            loss = self._compute_loss()
            loss.backward()
            self.opt_hyper.step()
            if self.opt_nn is not None:
                self.opt_nn.step()
            progress_bar.set_description(
                f"Iter: {i:02d} loss: {loss.item(): .2f}")
            if writer is not None:
                writer.add_scalar('loss', loss.item(), i)
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name, param.grad, i)
        self.eval()

    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Makes predictions.

        Args:
            x_test (np.ndarray): Test inputs of shape (num_inputs, dim_inputs).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing predictive mean
                and predictive standard deviation of shape (num_inputs, 1).

        """
        x_test_tensor = torch.as_tensor(x_test,
                                        dtype=self.dtype,
                                        device=self.device)
        mean_tensor, std_tensor = self.forward(x_test_tensor)
        return mean_tensor.cpu().numpy(), std_tensor.cpu().numpy()

    def forward(
        self,
        x_test: torch.Tensor,
        noise_free: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Make prediction.

        Args:
            x_test (torch.Tensor): Test inputs of shape
                (num_inputs, dim_inputs).
            noise_free (bool, optional): If True, predict the latent function
                values. Otherwise, predict the noisy targets.

        Returns:
            mean (torch.Tensor): Predictive mean of shape (num_inputs, 1)
            std (torch.Tensor): Predictive standard deviation of shape
                (num_inputs, 1).

        ??? note "Difference between `forward` and `predict`"

            The `forward` method uses PyTorch tensors as inputs and outputs.
            The `predict` method uses NumPy arrays as inputs and outputs.
            Users should use `predict` method for model prediction.

        """
        with torch.no_grad():
            L, iK_y = self._compute_common()
            Ksn = self.kernel(x_test, self.x_train)
            Kss_diag = self.kernel.diag(x_test)
            iL_Kns = torch.linalg.solve_triangular(L, Ksn.t(), upper=False)
            mean = Ksn @ iK_y
            var = Kss_diag - iL_Kns.square().sum(0).view(-1, 1)
            # Variance might be zero when lengthscale is too large.
            if torch.any(var <= 0.0):
                print(var.ravel().numpy())
                raise ValueError("Predictive variance <= 0.0!")
            var.clamp_(min=self.jitter)
            if not noise_free:
                var += self.noise
            std = var.sqrt()
        return mean, std

    @property
    def noise(self) -> torch.Tensor:
        r"""The noise variance hyper-parameter.

        Returns:
            torch.Tensor: The noise variance of the Gaussian likelihood.

        ??? note "Positivity"

            The positivity of noise variance is ensured by a softplus function.

        """
        return torch_utils.softplus(self._free_noise)

    @noise.setter
    def noise(self, noise: torch.Tensor) -> None:
        r"""The noise variance hyper-parameter.

        Args:
            noise (torch.Tensor): The new value of the noise variance.

        """
        with torch.inference_mode():
            self._free_noise.copy_(torch_utils.inv_softplus(noise))

    def _add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        r"""Add new data to the training set.

        Args:
            x_new (np.ndarray): New training inputs of shape
                (num_inputs, dim_inputs).
            y_new (np.ndarray): New training outputs of shape
                (num_outputs, dim_outputs).

        """
        self._validate_data(x_new, y_new)
        if not (hasattr(self, 'x_train') and hasattr(self, 'y_train')):
            self.x_train = torch.as_tensor(x_new,
                                           dtype=self.dtype,
                                           device=self.device)
            self.y_train = torch.as_tensor(y_new,
                                           dtype=self.dtype,
                                           device=self.device)
        else:
            self.x_train = torch.cat((self.x_train,
                                      torch.as_tensor(
                                          x_new,
                                          dtype=self.dtype,
                                          device=self.device,
                                      )))
            self.y_train = torch.cat((self.y_train,
                                      torch.as_tensor(
                                          y_new,
                                          dtype=self.dtype,
                                          device=self.device,
                                      )))

    def _validate_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        r"""Check if the inputs `x_new` and `y_new` are valid.

        Args:
            x_new (np.ndarray): An array of shape (num_inputs, dim_inputs)
                containing the input features of the new data.
            y_new (np.ndarray): An array of shape (num_outputs, 1)
                containing the output targets of the new data.

        Raises:
            ValueError: If any of the following conditions are met:
                1. `x_new` is not 2D.
                2. `y_new` is not 2D.
                3. `y_new` has more than 1 column.
                4. `x_new` and `y_new` have different number of samples.
                5. `x_new` and `self.x_train` have different number of features.
                6. `y_new` and `self.y_train` have different number of columns.

        """
        if x_new.ndim != 2:
            raise ValueError("x_train must be 2D.")
        if y_new.ndim != 2:
            raise ValueError("y_train must be 2D.")
        if y_new.shape[1] != 1:
            raise ValueError("Only support univariate output for now.")
        if x_new.shape[0] != y_new.shape[0]:
            raise ValueError("x_train and y_train should have same length.")
        if (hasattr(self, 'x_train')
                and x_new.shape[1] != self.x_train.shape[1]):
            raise ValueError("x_train and x_new should have same shape.")
        if (hasattr(self, 'y_train')
                and y_new.shape[1] != self.y_train.shape[1]):
            raise ValueError("y_train and y_new should have same shape.")

    def _compute_loss(self) -> torch.Tensor:
        r"""Compute training loss.

        Returns:
            torch.Tensor: The training loss.

        ??? note "Loss Function: Negative Log Marginal Likelihood"

            $$
            -\log{p(\mathbf{y}|X)}=
            \frac{1}{2}\mathbf{y}^{\intercal}\mathbf{K}_{y}\mathbf{y}
            +\frac{1}{2}\log{|\mathbf{K}_{y}|}
            +\frac{n}{2}\log{2\pi}.
            $$

        """
        L, iK_y = self._compute_common()
        quadratic = torch.sum(self.y_train * iK_y)
        logdet = L.diag().square().log().sum()
        constant = len(self.y_train) * np.log(2 * np.pi)
        return 0.5 * (quadratic + logdet + constant)

    def _compute_common(self):
        r"""Compute common terms for `_compute_loss` and `predict`.

        Returns:
            L (torch.Tensor): Lower Cholesky factor of the training covariance
                matrix. Shape: (num_train, num_train).
            iK_y (torch.Tensor): Inverse training covariance matrix multiplied
                by training outputs. Shape: (num_train, 1).

        ??? note "What are L and iK_y?"

            $$
            \boldsymbol{K}_{y} = \boldsymbol{K} + \sigma^2 \mathbf{I}
            = \boldsymbol{L}\boldsymbol{L}^{\intercal}
            $$

            $$
            \boldsymbol{K}_{y}^{-1} \mathbf{y} =
            \mathtt{cholesky solve}(\boldsymbol{y}, \boldsymbol{L})
            $$
        """
        K = self.kernel(self.x_train, self.x_train)
        K.diagonal().add_(self.noise)
        L = torch_utils.robust_cholesky(K, jitter=self.jitter)
        iK_y = torch.cholesky_solve(self.y_train, L, upper=False)
        return L, iK_y

    def _init_optimizers(self, lr_hyper: float, lr_nn: float) -> None:
        """Initialize optimizers for hyper-parameters and, optinally,
        neural network parameters in non-stationary kernels.

        Args:
            lr_hyper (float, optional): Learning rate of hyper-parameters.
                Defaults to 0.01.
            lr_nn (float, optional): Learning rate of neural network parameters
                in non-stationary kernels. Defaults to 0.001.

        !!! note "Neural Network Parameters"

            Neural network parameters are found by searching for the string
            "nn" in the parameter name.

        """
        self.lr_hyper, self.lr_nn = lr_hyper, lr_nn
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
