import torch
import torch.nn.functional as F


def softplus(x):
    """Transform the input to positive output."""
    return F.softplus(x, 1.0, 20.0) + 1e-6


def inv_softplus(y):
    """Inverse softplus function."""
    if torch.any(y <= 0.0):
        raise ValueError("Input to `inv_softplus` must be positive.")
    _y = y - 1e-6
    return _y + torch.log(-torch.expm1(-_y))


def constraint(free_parameter):
    """Returns constraint parameter."""
    return softplus(free_parameter)


def unconstraint(parameter: float):
    """Returns unconstraint tensor."""
    return inv_softplus(torch.tensor(
        parameter,
        dtype=torch.float64,
    ))


def robust_cholesky(cov_mat, jitter: float = 1e-6, num_attempts: int = 3):
    """Numerically stable Cholesky.

    Parameters
    ----------
    cov_mat: TensorType["num_samples", "num_samples"]
        Covariance matrix to be decomposed.
    jitter: float = 1e-6
        Small positive number added to the digonal elements of covariance
        matrix for preventing Cholesky decomposition failure.
    num_tries: int = 3
        Number of attempts (with successively increasing jitter) to make before
        raising an error.

    Notes
    -----
    This code snippet is adapted from the GPyTorch library:
    https://github.com/cornellius-gp/gpytorch/blob/5a0ff6b59720b3db1cbaf0063bd10486d2d4213e/gpytorch/utils/cholesky.py#L12

    """
    L, info = torch.linalg.cholesky_ex(cov_mat, out=None)
    if not torch.any(info):
        return L
    if torch.any(torch.isnan(cov_mat)):
        raise ValueError("Encountered NaN in cov_mat.")
    _cov_mat = cov_mat.clone()
    jitter_prev = 0.0
    jitter_new = jitter
    for i in range(num_attempts):
        is_positive_definite = info > 0
        jitter_new = jitter * (10**i)
        increment = is_positive_definite * (jitter_new - jitter_prev)
        _cov_mat.diagonal().add_(increment)
        jitter_prev = jitter_new
        print("Matrix is not positive definite! " +
              f"Added {jitter_new:.1e} to the diagonal.")
        L, info = torch.linalg.cholesky_ex(_cov_mat, out=None)
        if not torch.any(info):
            return L
    raise ValueError(
        "Covariance matrix is still not positive-definite " +
        f"after adding {jitter_new:.1e} to the diagonal elements.")
