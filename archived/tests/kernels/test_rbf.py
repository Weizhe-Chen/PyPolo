import torch


def test_init(rbf):
    expected_amplitude = torch.tensor(1.0, dtype=torch.float64)
    torch.testing.assert_close(rbf.amplitude, expected_amplitude)
    expected_lengthscale = torch.tensor(1.0, dtype=torch.float64)
    torch.testing.assert_close(rbf.lengthscale, expected_lengthscale)


def test_normal_rbf_self_covariance_matrix(rbf):
    x_1 = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_1)
    expected_cov_mat = torch.tensor(
        [
            [0.0, -4.0],
            [-4.0, 0],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)


def test_normal_rbf_cross_covariance_matrix(rbf):
    x_1 = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.float64,
    )
    x_2 = torch.tensor(
        [
            [1, 3],
            [2, 4],
            [9, 9],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_2)
    expected_cov_mat = torch.tensor(
        [
            [-1.0 / 2.0, -5.0 / 2.0, -113.0 / 2.0],
            [-5.0 / 2.0, -1.0 / 2.0, -61.0 / 2.0],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)


def test_small_lengthscale_rbf_self_covariance_matrix(rbf):
    rbf.lengthscale = 1e-4
    x_1 = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_1)
    expected_cov_mat = torch.tensor(
        [
            [0.0, -2 / 1e-8],
            [-2 / 1e-8, 0.0],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)


def test_small_lengthscale_rbf_cross_covariance_matrix(rbf):
    rbf.lengthscale = 1e-4
    x_1 = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.float64,
    )
    x_2 = torch.tensor(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_2)
    expected_cov_mat = torch.tensor(
        [
            [-4 / 1e-8, -2 / 1e-8],
            [-2 / 1e-8, -4 / 1e-8],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)


def test_large_lengthscale_rbf_self_covariance_matrix(rbf):
    rbf.lengthscale = 1e4
    x_1 = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_1)
    expected_cov_mat = torch.tensor(
        [
            [0.0, -2 / 1e8],
            [-2 / 1e8, 0.0],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)


def test_large_lengthscale_rbf_cross_covariance_matrix(rbf):
    rbf.lengthscale = 1e4
    x_1 = torch.tensor(
        [
            [-1, -1],
            [-1, 1],
        ],
        dtype=torch.float64,
    )
    x_2 = torch.tensor(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=torch.float64,
    )
    actual_cov_mat = rbf(x_1, x_2)
    expected_cov_mat = torch.tensor(
        [
            [-4 / 1e8, -2 / 1e8],
            [-2 / 1e8, -4 / 1e8],
        ],
        dtype=torch.float64,
    ).exp_()
    torch.testing.assert_allclose(actual_cov_mat, expected_cov_mat)
