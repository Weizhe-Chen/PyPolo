import numpy as np


def gaussian_entropy(std: np.ndarray) -> np.ndarray:
    """Compute the entropy of a Gaussian distribution given standard deviation.

    Parameters
    ----------
    std : np.ndarray
        Standard deviation array.

    Returns
    -------
    entropy: np.ndarray
        Entropy array.

    """
    entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    return entropy
