import numpy as np
import pytest
import torch

from pypolo.models import GPRModel
from pypolo.models.kernels import GaussianKernel
from matplotlib import pyplot as plt


@pytest.fixture
def model():
    np.random.seed(123)
    torch.manual_seed(123)
    kernel = GaussianKernel(lengthscale=3.0, amplitude=1.0, device_name="cpu")
    model = GPRModel(device_name="cpu", kernel=kernel, noise=0.01)
    return model


def test_predict(model, verbose, render):
    # Generate some random training data
    X_train = np.random.uniform(-5, 5, size=(20, 1))
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, size=(20, 1))

    # Train the model on the data
    model.learn(X_train, y_train, num_iter=500, verbose=verbose)

    # Generate some test data
    X_test = np.linspace(-5, 5, num=100).reshape(-1, 1)

    # Make predictions
    y_pred, y_std = model.predict(X_test)

    # Check that the predictions have the correct shape
    assert y_pred.shape == (100, 1)
    assert y_std.shape == (100, 1)

    # Check that the mean prediction is close to the true function values
    rmse = np.sqrt(np.mean((y_pred.squeeze() - np.sin(X_test.squeeze()))**2))
    assert rmse < (y_std.mean() + 0.01)

    # Check that the standard deviation is non-negative
    assert np.all(y_std >= 0)

    if render:
        # Plot the results
        plt.figure()
        plt.plot(X_train, y_train, "kx", label="Training data")
        plt.plot(X_test, np.sin(X_test), "b", label="True function")
        plt.plot(X_test, y_pred, "r", label="Predicted mean")
        plt.fill_between(
            X_test.squeeze(),
            (y_pred - 2 * y_std).squeeze(),
            (y_pred + 2 * y_std).squeeze(),
            color="r",
            alpha=0.2,
            label="Predicted uncertainty",
        )
        plt.legend(loc="upper left")
        plt.show()
