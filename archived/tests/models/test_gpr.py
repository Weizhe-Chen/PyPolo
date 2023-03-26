from matplotlib import pyplot as plt
import numpy as np
import pytest
import torch

from pypolo.utilities.linalg import inv_softplus


def test_init(rbf_gpr):
    expected_free_params = inv_softplus(
        torch.tensor(
            [
                0.01,  # noise variance
                1.0,  # amplitude
                0.2,  # lengthscale
            ],
            dtype=torch.float64,
        ))
    for expected, actual in zip(expected_free_params, rbf_gpr.parameters()):
        torch.testing.assert_close(expected, actual)
    expected_params = {
        'lr': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0,
        'amsgrad': False,
        'params': [0, 1, 2],
    }
    assert rbf_gpr.opt_hyper.state_dict()["param_groups"][0] == expected_params
    assert rbf_gpr.opt_nn is None
    np.testing.assert_almost_equal(actual=rbf_gpr.jitter, desired=1e-6)


def test_check_shape_no_exception(rbf_gpr):
    x_train = np.array([
        [0, 1],
        [2, 3],
    ], dtype=np.float64)
    y_train = np.array([
        [0.1],
        [0.2],
    ], dtype=np.float64)
    try:
        rbf_gpr.check_shape(x_train, y_train)
    except ValueError as e:
        pytest.fail(f"check_shape raised an unexpected exception {e}.")


def test_check_shape_with_exception(rbf_gpr):
    # x_train should be a column vector.
    with pytest.raises(ValueError):
        x_train = np.array([0, 2], dtype=np.float64)
        y_train = np.array([0.1, 0.2], dtype=np.float64).reshape(-1, 1)
        rbf_gpr.check_shape(x_train, y_train)
    # y_train should be a column vector.
    with pytest.raises(ValueError):
        x_train = np.array([0, 2], dtype=np.float64).reshape(-1, 1)
        y_train = np.array([0.1, 0.2], dtype=np.float64)
        rbf_gpr.check_shape(x_train, y_train)
    # x_train and y_train both should be column vector.
    with pytest.raises(ValueError):
        x_train = np.array([0, 2], dtype=np.float64)
        y_train = np.array([0.1, 0.2], dtype=np.float64)
        rbf_gpr.check_shape(x_train, y_train)
    # y_train should be one-dimensional column vector.
    with pytest.raises(ValueError):
        x_train = np.array([0, 2], dtype=np.float64).reshape(-1, 1)
        y_train = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ], dtype=np.float64)
        rbf_gpr.check_shape(x_train, y_train)
    # x_train and y_train should have same length.
    with pytest.raises(ValueError):
        x_train = np.array([0.1, 0.2, 0.3], dtype=np.float64).reshape(-1, 1)
        y_train = np.array([0.1, 0.2], dtype=np.float64).reshape(-1, 1)
        rbf_gpr.check_shape(x_train, y_train)


def plot_1d_prediction(x, f, y, mean, std):
    _, ax = plt.subplots()
    ax.scatter(x, y, color='k', s=10)
    ax.plot(x, f, 'r--')
    ax.plot(x, mean, 'b')
    ax.fill_between(
        x.ravel(),
        (mean - 2.0 * std).ravel(),
        (mean + 2.0 * std).ravel(),
        color='b',
        alpha=0.3,
    )
    ax.set_xlim([x.min(), x.max()])
    plt.show()


def compute_mean_negative_log_predictive_density(error, std):
    return np.mean(0.5 * np.log(2 * np.pi) + np.log(std) +
                   0.5 * np.square(error / std))


def compute_root_mean_squared_error(error):
    return np.sqrt(np.mean(np.square(error)))


def test_forward(rbf_gpr, gpr_data):
    x, f, _ = gpr_data
    mean, std = rbf_gpr(x)
    error = mean - f
    nlpd = compute_mean_negative_log_predictive_density(error, std)
    rmse = compute_root_mean_squared_error(error)
    assert rmse < 0.1, "GPR prediction has large rmse."
    assert nlpd < -1.2, "GPR prediction has large nlpd."
    #  # Uncomment the following line for visualization
    #  plot_1d_prediction(x, f, y, mean, std)


def test_loss_and_optimize(rbf_gpr, gpr_data):
    verbose = False
    num_iter = 10
    x, f, _ = gpr_data
    with torch.no_grad():
        expected_noise = rbf_gpr.noise.item()
        expected_amplitude = rbf_gpr.kernel.amplitude.item()
        expected_lengthscale = rbf_gpr.kernel.lengthscale.item()
    if verbose:
        print("\nHyper-parameters used to generate the data:")
        print(f"Noise var:\t{expected_noise:.2f}")
        print(f"Amplitude:\t{expected_amplitude:.2f}")
        print(f"Lengthscale:\t{expected_lengthscale:.2f}")
    rbf_gpr.optimize(num_iter=num_iter, verbose=verbose)
    with torch.no_grad():
        actual_noise = rbf_gpr.noise.item()
        actual_amplitude = rbf_gpr.kernel.amplitude.item()
        actual_lengthscale = rbf_gpr.kernel.lengthscale.item()
    if verbose:
        print("Hyper-parameters after optimization:")
        print(f"Noise var:\t{actual_noise:.2f}")
        print(f"Amplitude:\t{actual_amplitude:.2f}")
        print(f"Lengthscale:\t{actual_lengthscale:.2f}")
    assert np.fabs(expected_noise - actual_noise) < 0.1
    assert np.fabs(expected_amplitude - actual_amplitude) < 0.1
    assert np.fabs(expected_lengthscale - actual_lengthscale) < 0.1
    mean, std = rbf_gpr(x)
    error = mean - f
    nlpd = compute_mean_negative_log_predictive_density(error, std)
    rmse = compute_root_mean_squared_error(error)
    assert rmse < 0.1, "GPR prediction has large rmse."
    assert nlpd < -1.2, "GPR prediction has large nlpd."
    #  # Uncomment the following line for visualization
    #  plot_1d_prediction(x, f, y, mean, std)
