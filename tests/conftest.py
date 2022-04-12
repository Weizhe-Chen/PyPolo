import numpy as np
import pytest
import torch

from pypolo.dynamics import DubinsCar
from pypolo.utilities import GridMap
from pypolo.kernels import RBF
from pypolo.models import GPR
from pypolo.utilities.linalg import robust_cholesky


@pytest.fixture(scope="module")
def dubins_car() -> DubinsCar:
    return DubinsCar(rate=10.0)


@pytest.fixture(scope="module")
def grid_map() -> GridMap:
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    return GridMap(matrix=env, extent=extent)

@pytest.fixture(scope="module")
def rbf() -> RBF:
    return RBF(amplitude=1.0, lengthscale=1.0)

@pytest.fixture(scope="module")
def gpr_data():
    amplitude = 1.0
    lengthscale = 0.2
    noise_scale = 0.1
    num_samples = 300
    xmin = 0.0
    xmax = 1.0
    jitter = 1e-5
    torch.manual_seed(1)
    rbf = RBF(amplitude=amplitude, lengthscale=lengthscale)
    x = np.linspace(xmin, xmax, num_samples).reshape(-1, 1)
    _x = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        cov_mat = rbf(_x, _x)
        cov_mat.diagonal().add_(jitter)
        cholesky = robust_cholesky(cov_mat)
        f = cholesky @ torch.randn(num_samples, 1, dtype=torch.float64)
        y = f + torch.randn(num_samples, 1, dtype=torch.float64) * noise_scale
    f = f.numpy()
    y = y.numpy()
    return x, f, y

@pytest.fixture(scope="module")
def rbf_gpr(gpr_data) -> GPR:
    x, _, y = gpr_data
    kernel = RBF(amplitude=1.0, lengthscale=0.2)
    noise = 0.01
    lr_hyper = 0.01
    lr_nn = 0.001
    jitter = 1e-6
    rbf_gpr = GPR(x, y, kernel, noise, lr_hyper, lr_nn, jitter)
    return rbf_gpr
