from typing import Tuple

import numpy as np

from ..dynamics import DubinsCar
from .robot import IRobot


class USV(IRobot):
    """Unmanned Surface Vehicle."""
    def __init__(
        self,
        init_state: np.ndarray,
        control_rate: float,
        max_lin_vel: float,
        tolerance: float,
        sampling_rate: float,
    ) -> None:
        """

        Parameters
        ----------
        init_state: np.ndarray, shape=(dim_states, ), dtype=np.float64
            Initial robot state.
        control_rate: float
            Control update rate (Hz).
        max_lin_vel: float
            Maximum linear velocity (m/s).
        tolerance: float
            Tolerance of error from the current state to the goal state.
        sampling_rate: float
            Sampling rate.

        """
        self._check_inputs(
            init_state,
            control_rate,
            max_lin_vel,
            tolerance,
        )
        dynamics = DubinsCar(control_rate)
        super().__init__(init_state, dynamics, tolerance, sampling_rate)
        self.max_lin_vel = max_lin_vel

    @staticmethod
    def _check_inputs(
        init_state: np.ndarray,
        control_rate: float,
        max_lin_vel: float,
        tolerance: float,
    ):
        """

        Parameters
        ----------
        init_state: np.ndarray, shape=(dim_states, ), dtype=np.float64
            Initial robot state.
        control_rate: floata, positive
            Control update rate (Hz).
        max_lin_vel: float, positive
            Maximum linear velocity (m/s).
        tolerance: float, positive
            Tolerance of error from the current state to the goal state.

        """
        if init_state.ndim != 1 or init_state.dtype != np.float64:
            raise ValueError("init_state: np.ndarray, " +
                             "shape=(dim_states, ), dtype=np.float64")
        if control_rate <= 0.0:
            raise ValueError("control_rate: float, positive, Hz")
        if max_lin_vel <= 0.0:
            raise ValueError("max_lin_vel: float, positive, m/s")
        if tolerance <= 0.0:
            raise ValueError("tolerance: float, positive, m")

    def control(self) -> Tuple[float, np.ndarray]:
        """Compute control output, i.e. action.

        Returns
        -------
        dist: float
            Distance to the first goal state.
        action: np.ndarray
            Control output.

        """
        assert self.has_goal, "I need at least one goal state do control."

        x, y, o = self.state

        # Compute distance to the goal.
        goal_state = self.goal_states[0]
        goal_x, goal_y = goal_state[:2]
        x_diff = goal_x - x
        y_diff = goal_y - y
        dist = np.hypot(x_diff, y_diff)

        # Compute the goal position in the odometry frame.
        x_odom = np.cos(o) * x_diff + np.sin(o) * y_diff
        y_odom = -np.sin(o) * x_diff + np.cos(o) * y_diff

        linear_velocity = self.max_lin_vel * np.tanh(x_odom)
        # angular proportional parameter is set to 2.0
        angular_velocity = 2.0 * np.arctan2(y_odom, x_odom)

        action = np.array([linear_velocity, angular_velocity])
        return dist, action
