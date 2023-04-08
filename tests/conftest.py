import os

import pytest
import torch


@pytest.fixture(scope="module")
def verbose() -> bool:
    return False


@pytest.fixture(scope="module")
def render() -> bool:
    if os.getenv('CI'):
        print('Disabling rendering in CI environment.')
        return False
    else:
        return True


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")
