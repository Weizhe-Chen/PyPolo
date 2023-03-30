from typing import Callable

import numpy as np


def gaussian_entropy(predict_fn: Callable, x: np.ndarray) -> np.ndarray:
    r"""Calculates the element-wise entropy of a Gaussian predictive
        distribution at the given inputs.

    !!! note "Entropy of the univariate Gaussian"

        $$
        x\sim\mathcal{N}(\mu, \sigma^{2})
        $$

        $$
        H(x)=\frac{1}{2}\log(2\pi\sigma^2)+\frac{1}{2}
        $$

    Args:
        x (np.ndarray): Input array of shape (num_inputs, dim_inputs) to be
            evaluated.

    Returns:
        np.ndarray: An array of entropy values corresponding to the inputs.

    """
    _, std = predict_fn(x)
    entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    return entropy
