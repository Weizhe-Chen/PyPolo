import numpy as np

from ..utils import TensorMap
from .base_sensor import BaseSensor


class EnvironmentalSensor(BaseSensor):

    def __init__(self, rate: float, noise_scale: float) -> None:
        r"""Initialize the sensor.

        Args:
            rate (float): sensing rate (Hz).
            noise_scale (float): Standard deviation of the Gaussian noise.

        """
        super().__init__(rate)
        self.noise_scale = noise_scale

    def sense(
        self,
        robot_state: np.ndarray,
        env_state: TensorMap,
    ) -> np.ndarray:
        r"""Sense the environment.

        Args:
            robot_state (np.ndarray): robot state.
            env_state (TensorMap): environment state.

        Returns:
            np.ndarray: sensor output of shape (num_obs, ).

        """
        ground_truth = env_state.get_values(robot_state[0], robot_state[1])
        observation = ground_truth + np.random.normal(0, self.noise_scale)
        assert observation.shape == (len(ground_truth), )
        return observation
