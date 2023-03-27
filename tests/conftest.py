import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")
