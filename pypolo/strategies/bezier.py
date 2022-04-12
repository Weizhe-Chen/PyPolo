from bezier import Curve
import numpy as np
from .strategy import IStrategy

from ..models import IModel
from typing import List


class Bezier(IStrategy):
    """Bezier curve pilot data collector"""
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
    ) -> None:
        """

        Parameters
        ----------
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Bounding box of the sampling task workspace.
        rng: np.random.RandomState, unused
            Random number generator if `get` has random operations.

        """
        super().__init__(task_extent, rng)
        self.generate_bezier_curve(self.generate_control_points())

    def generate_control_points(self):
        """Generate control points

        Returns
        -------
        control_points: np.ndarray, shape=(2, 18)
            Control points of the Bezier curve.

        """
        xmin, xmax, ymin, ymax = self.task_extent
        xhalf = (xmax - xmin) / 2 + xmin
        xquater = (xmax - xmin) / 4 + xmin
        xquater_x3 = 3 * (xmax - xmin) / 4 + xmin
        yhalf = (ymax - ymin) / 2 + ymin
        yquater = (ymax - ymin) / 4 + ymin
        yquater_x3 = 3 * (ymax - ymin) / 4 + ymin

        control_points = np.array([
            [xquater, ymin],
            [xmin, ymin],
            [xmin, yquater],
            [xmin, yhalf],
            [xmin, yquater_x3],
            [xmin, ymax],
            [xquater, ymax],
            [xhalf, ymax],
            [xquater_x3, ymax],
            [xmax, ymax],
            [xmax, yquater_x3],
            [xmax, yhalf],
            [xmax, yquater],
            [xmax, ymin],
            [xquater_x3, ymin],
            [xhalf, ymin],
            [xhalf, yquater],
            [xhalf, yhalf],
        ]).T
        return control_points

    def generate_bezier_curve(self, control_points: np.ndarray):
        """Generate Bezier curve.

        Parameters
        ----------
        control_points: np.ndarray, shape=(2, 18)
            Control points of the Bezier curve.

        Attributes
        ----------
        curve: bezier.Curve
            Bezier curve object.

        """
        nodes = np.asfortranarray(control_points)
        self.curve = Curve(nodes, degree=nodes.shape[1] - 1)

    def get(self, num_states: int, model=None) -> np.ndarray:
        """Get goal states for sampling.

        Parameters
        ----------
        num_states: int,
            Number of goal states.
        model: IModel, unused
            A probabilistic model that provides `mean` and `std` via `forward`.

        Returns
        -------
        goal_states: np.ndarray, shape=(num_states, dim_states)
            Sampling goal states.

        """
        curve_params = np.linspace(0, 1, num_states)
        waypoints = self.curve.evaluate_multi(curve_params).T
        return waypoints
