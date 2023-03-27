import pytest
import torch

from pypolo.models.kernels import GaussianKernel


@pytest.fixture(scope="module")
def isotropic() -> GaussianKernel:
    return GaussianKernel(lengthscale=1.0, amplitude=1.0, device_name="cpu")


def test_gaussian_kernel_init(isotropic: GaussianKernel, device: torch.device):
    assert isotropic.device == device
    assert isotropic.dtype == torch.double
    assert isotropic._free_amplitude.requires_grad
    assert isotropic._free_amplitude.dtype == torch.double
    assert isotropic._free_amplitude.item() == pytest.approx(0.54132327)
    assert isotropic._free_lengthscale.requires_grad
    assert isotropic._free_lengthscale.dtype == torch.double
    assert isotropic._free_lengthscale.item() == pytest.approx(0.54132327)


def test_gaussian_kernel_amplitude_getter(isotropic: GaussianKernel):
    assert isinstance(isotropic.amplitude, torch.Tensor)
    assert isotropic.amplitude.item() == pytest.approx(1.0)
    assert isotropic.amplitude.dtype == torch.double
    assert isotropic.amplitude.requires_grad


def test_gaussian_kernel_lengthscale_getter(isotropic: GaussianKernel):
    assert isinstance(isotropic.lengthscale, torch.Tensor)
    assert isotropic.lengthscale.item() == pytest.approx(1.0)
    assert isotropic.lengthscale.dtype == torch.double
    assert isotropic.lengthscale.requires_grad


def test_gaussian_kernel_amplitude_setter(isotropic: GaussianKernel,
                                          device: torch.device):
    isotropic.amplitude = torch.tensor(2.0, dtype=torch.double, device=device)
    assert isotropic.amplitude.item() == pytest.approx(2.0)
    assert isotropic.amplitude.dtype == torch.double
    assert isotropic.amplitude.requires_grad


def test_gaussian_kernel_lengthscale_setter(isotropic: GaussianKernel,
                                            device: torch.device):
    isotropic.lengthscale = torch.tensor(2.0,
                                         dtype=torch.double,
                                         device=device)
    assert isotropic.lengthscale.item() == pytest.approx(2.0)
    assert isotropic.lengthscale.dtype == torch.double
    assert isotropic.lengthscale.requires_grad


def test_gaussian_kernel_diag(isotropic: GaussianKernel, device: torch.device):
    isotropic.amplitude = torch.tensor(1.0, dtype=torch.double, device=device)
    x = torch.randn((2, 3), dtype=torch.double, device=device)
    diag = isotropic.diag(x)
    assert isinstance(diag, torch.Tensor)
    assert diag.shape == (2, 1)
    assert diag.dtype == torch.double
    assert diag.requires_grad
    assert diag[0].item() == pytest.approx(1.0)
    assert diag[1].item() == pytest.approx(1.0)


def test_gaussian_kernel_forward(isotropic: GaussianKernel,
                                 device: torch.device):
    isotropic.amplitude = torch.tensor(1.0, dtype=torch.double, device=device)
    isotropic.lengthscale = torch.tensor(1.0,
                                         dtype=torch.double,
                                         device=device)
    x1 = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.double,
        device=device,
    )
    x_2 = torch.tensor(
        [
            [1, 3],
            [2, 4],
            [9, 9],
        ],
        dtype=torch.float64,
        device=device,
    )

    # Test self covariance
    expected = torch.tensor(
        [
            [0.0, -4.0],
            [-4.0, 0.0],
        ],
        dtype=torch.double,
        device=device,
    ).exp_()
    assert torch.allclose(isotropic(x1, x1), expected)

    # Test cross covariance
    expected = torch.tensor(
        [
            [-1.0 / 2.0, -5.0 / 2.0, -113.0 / 2.0],
            [-5.0 / 2.0, -1.0 / 2.0, -61.0 / 2.0],
        ],
        dtype=torch.float64,
    ).exp_()
    assert torch.allclose(isotropic(x1, x_2), expected)

    # Test small-lengthscale stability
    isotropic.lengthscale = torch.tensor(1e-4,
                                         dtype=torch.double,
                                         device=device)
    x3 = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.float64,
        device=device,
    )
    x4 = torch.tensor(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=torch.float64,
        device=device,
    )
    expected = torch.tensor(
        [
            [0.0, -2 / 1e-8],
            [-2 / 1e-8, 0.0],
        ],
        dtype=torch.float64,
        device=device,
    ).exp_()
    assert torch.allclose(isotropic(x3, x3), expected)
    expected = torch.tensor(
        [
            [-4 / 1e-8, -2 / 1e-8],
            [-2 / 1e-8, -4 / 1e-8],
        ],
        dtype=torch.float64,
        device=device,
    ).exp_()
    assert torch.allclose(isotropic(x3, x4), expected)

    # Test large-lengthscale stability
    isotropic.lengthscale = torch.tensor(1e4,
                                         dtype=torch.double,
                                         device=device)
    expected = torch.tensor(
        [
            [0.0, -2 / 1e8],
            [-2 / 1e8, 0.0],
        ],
        dtype=torch.float64,
        device=device,
    ).exp_()
    assert torch.allclose(isotropic(x3, x3), expected)

    expected = torch.tensor(
        [
            [-4 / 1e8, -2 / 1e8],
            [-2 / 1e8, -4 / 1e8],
        ],
        dtype=torch.float64,
        device=device,
    ).exp_()
    assert torch.allclose(isotropic(x3, x4), expected)
