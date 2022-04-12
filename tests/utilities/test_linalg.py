import pytest
import torch

from pypolo.utilities import linalg


def test_softplus():
    torch.manual_seed(0)
    offset = 1e4
    epsilon = 1e-6
    # Normal input
    x = torch.randn(10, dtype=torch.float64)
    expected = torch.log(1 + torch.exp(x)) + epsilon
    actual = linalg.softplus(x)
    torch.testing.assert_allclose(actual, expected)
    # Extremely negative input
    x = torch.randn(10, dtype=torch.float64) - offset
    expected = torch.zeros_like(x) + epsilon
    actual = linalg.softplus(x)
    torch.testing.assert_allclose(actual, expected)
    # Extremely positive input
    x = torch.randn(10, dtype=torch.float64) + offset
    actual = linalg.softplus(x)
    torch.testing.assert_allclose(actual, x + epsilon)


def test_inv_softplus():
    torch.manual_seed(0)
    offset = 1e4
    # Normal case
    expected = torch.rand(10)
    softplused = linalg.softplus(expected)
    actual = linalg.inv_softplus(softplused)
    torch.testing.assert_allclose(actual, expected)
    # Extremely large positive input
    expected = torch.randn(10, dtype=torch.float64) + offset
    softplused = linalg.softplus(expected)
    actual = linalg.inv_softplus(softplused)
    torch.testing.assert_allclose(actual, expected)
    # Extremely small positive input
    expected = torch.randn(10, dtype=torch.float64) * 1e-4
    softplused = linalg.softplus(expected)
    actual = linalg.inv_softplus(softplused)
    torch.testing.assert_allclose(actual, expected)


def test_inv_softplus_with_exception():
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        invalid_input = torch.tensor(-0.1, dtype=torch.float64)
        linalg.inv_softplus(invalid_input)


def test_constraint():
    free_parameter = torch.tensor(-1.0, dtype=torch.float64)
    actual = linalg.constraint(free_parameter)
    expected = torch.log(torch.exp(free_parameter) + 1) + 1e-6
    torch.testing.assert_close(actual, expected)

    free_parameter = torch.tensor(1.0, dtype=torch.float64)
    actual = linalg.constraint(free_parameter)
    expected = torch.log(torch.exp(free_parameter) + 1) + 1e-6
    torch.testing.assert_close(actual, expected)

    free_parameter = torch.tensor(30.0, dtype=torch.float64)
    actual = linalg.constraint(free_parameter)
    expected = free_parameter + 1e-6
    torch.testing.assert_close(actual, expected)

    free_parameter = torch.tensor(-30.0, dtype=torch.float64)
    actual = linalg.constraint(free_parameter)
    expected = torch.tensor(1e-6, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_unconstraint():
    actual = linalg.unconstraint(1e-6)
    expected = torch.log(torch.exp(torch.tensor(0.0, dtype=torch.float64)) - 1)
    torch.testing.assert_close(actual, expected)

    actual = linalg.unconstraint(30.0)
    expected = torch.tensor(30.0, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_robust_cholesky_success():
    expected = torch.tensor([[1, 0], [2, 1]], dtype=torch.float64)
    target_matrix = expected @ expected.T
    actual = linalg.robust_cholesky(target_matrix)
    torch.testing.assert_allclose(actual, expected)


def test_robust_cholesky_nan_with_exception():
    expected = torch.tensor([[1, 0], [2, 1]], dtype=torch.float64)
    target_matrix = expected @ expected.T
    target_matrix[0, 0] = torch.nan
    with pytest.raises(ValueError):
        linalg.robust_cholesky(target_matrix)
