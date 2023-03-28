import pytest
import torch


@pytest.fixture(scope="module")
def verbose() -> bool:
    return False


@pytest.fixture(scope="module")
def render() -> bool:
    return False


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")
