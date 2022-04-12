import numpy as np

from .dynamics import IDynamics


class DubinsCar(IDynamics):
    """Dubins car kinematics."""
    def __init__(self, rate: float) -> None:
        super().__init__(rate)

    @staticmethod
    def check_shape(
        state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """Check whether the given state and action have correct shape.

        Parameters
        ----------
        state : numpy.ndarray, shape=(dim_state,)
            The current state.
        action: numpy.ndarray, shape=(dim_action,)
            The action applied to the current state.

        """
        if state.shape != (3, ):
            raise ValueError("state.shape should be (3,): [x, y, o]")
        if action.shape != (2, ):
            raise ValueError("action.shape should be (2,): [v, w]")

    @staticmethod
    def check_dtype(
        state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """Check whether the given state and action have correct dtype.

        Parameters
        ----------
        state : numpy.ndarray, shape=(dim_state,)
            The current state.
        action: numpy.ndarray, shape=(dim_action,)
            The action applied to the current state.

        """
        if state.dtype != np.float64:
            raise TypeError("state.dtype should be np.float64")
        if action.dtype != np.float64:
            raise TypeError("action.dtype should be np.float64")

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi].

        Parameters
        ----------
        angle: float
            The angle before normalization.

        Returns
        -------
        angle: float
            The angle after normalization.

        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.check_shape(state, action)
        self.check_dtype(state, action)
        x, y, o = state
        v, w = action
        state[0] = x + v * np.cos(o) * self.dt
        state[1] = y + v * np.sin(o) * self.dt
        state[2] = o + w * self.dt
        state[2] = self.normalize_angle(state[2])
        return state
