import numpy as np
from pypolo.objectives.entropy import gaussian_entropy


def test_single_std():
    std = np.array([1.0])
    expected_entropy = np.array([1.4189385332046727])
    entropy = gaussian_entropy(std)
    np.testing.assert_allclose(entropy, expected_entropy)


def test_multiple_std():
    std = np.array([1.0, 2.0, 3.0])
    expected_entropy = np.array(
        [1.4189385332046727, 2.112085713764618, 2.5175508218727822])
    entropy = gaussian_entropy(std)
    np.testing.assert_allclose(entropy, expected_entropy)
