from abc import ABC, abstractmethod
import numpy as np
from ..utils import TensorMap
from typing import Any


class BaseSensor(ABC):
    """Base class for all sensors."""

    def __init__(self, rate: float) -> None:
        r"""Initialize the sensor.

        Args:
            rate (float): sensing rate (Hz).

        """
        assert rate > 0, "`rate` must be positive!"
        self.rate = rate
        self.dt = 1.0 / rate

    @abstractmethod
    def sense(self, robot_state: np.ndarray, env_state: TensorMap) -> Any:
        r"""Sense the environment.

        Args:
            robot_state (np.ndarray): robot state.
            env_state (TensorMap): environment state.

        Returns:
            Any: sensor output.

        """
        raise NotImplementedError
