import numpy as np


def gaussian_entropy(std: np.ndarray) -> np.ndarray:
    r"""Calculates the entropy of a Gaussian distribution given its standard
        deviation.

    !!! note "Entropy of the univariate Gaussian"

        $$
        x\sim\mathcal{N}(\mu, \sigma^{2})
        $$

        $$
        H(x)=\frac{1}{2}\log(2\pi\sigma^2)+\frac{1}{2}
        $$

    Args:
        std (np.ndarray): An array of standard deviations of the Gaussian
            distribution.

    Returns:
        np.ndarray: An array of entropy values corresponding to each standard
            deviation.

    """
    entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    return entropy
