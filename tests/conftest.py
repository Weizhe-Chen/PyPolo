import numpy as np
import pytest
import torch


from pypolo.dynamics import DubinsCar
@pytest.fixture(scope="module")
def dubins_car() -> DubinsCar:
    return DubinsCar(rate=10.0)


from pypolo.utilities import GridMap
@pytest.fixture(scope="module")
def grid_map() -> GridMap:
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    return GridMap(matrix=env, extent=extent)

from pypolo.kernels import RBF
@pytest.fixture(scope="module")
def rbf() -> RBF:
    return RBF(amplitude=1.0, lengthscale=1.0)

from pypolo.utilities.linalg import robust_cholesky
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

from pypolo.models import GPR
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

from pypolo.sensors import Sonar
@pytest.fixture(scope="module")
def sonar() -> Sonar:
    extent = [-10.0, 10.0, -10.0, 10.0]
    rows, cols = np.ogrid[0:6, 0:7]
    env = rows + cols
    return Sonar(rate=1.0, env=env, env_extent=extent, noise_scale=0.1)

from pypolo.robots import USV
@pytest.fixture(scope="module")
def usv() -> USV:
    init_state = np.array([0, 0, 0], dtype=np.float64)
    usv = USV(init_state=init_state,
              control_rate=10.0,
              max_lin_vel=1.0,
              tolerance=0.1,
              sampling_rate=1.0)
    return usv
