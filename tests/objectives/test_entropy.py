import numpy as np
from pypolo.objectives import entropy

def test_gaussian_entropy():
    std = np.random.rand(3, 1)
    expected = 0.5 * np.log(2 * np.pi * std ** 2) + 0.5
    actual = entropy.gaussian_entropy(std)
    np.testing.assert_allclose(actual, expected)
