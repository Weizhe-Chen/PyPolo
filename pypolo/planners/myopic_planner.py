from typing import Callable, List

import numpy as np

from .base_planner import BasePlanner


class MyopicPlanner(BasePlanner):

    def __init__(
        self,
        workspace: List[float],
        objective: Callable,
        num_candidates: int = 1000,
    ) -> None:
        r"""Interface of a planner.

        Args:
            workspace (List[float]): Bounding box [xmin, ymin, xmax, ymax]
                of the planner's workspace.
            objective (Callable): The objective function.
            num_candidates (int): The number of candidates to evaluate.

        """
        assert len(workspace) == 4, "Workspace = [xmin, ymin, xmax, ymax]."
        self.workspace = workspace
        self.objective = objective
        self.num_candidates = num_candidates

    def plan(self, state: np.ndarray) -> np.ndarray:
        """Plan informative waypoint(s) given the robot state.

        Args:
            state (np.ndarray): The robot state.

        Returns:
            waypoints (np.ndarray): The resulting informative waypoint(s) of
                shape (num_waypoints, 2).

        """
        # Sample candidates.
        candidates = np.random.uniform(low=self.workspace[:2],
                                       high=self.workspace[2:],
                                       size=(self.num_candidates, 2))
        # Evaluate candidates.
        informativeness = self.objective(candidates)
        diffs = candidates - state[:2]
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        # Normalize informativeness and distance.
        informativeness = ((informativeness - informativeness.min()) /
                           informativeness.ptp())
        dists = (dists - dists.min()) / dists.ptp()
        # Compute scores.
        scores = informativeness - dists
        # Select the candidate with the highest score.
        waypoint = candidates[np.argmax(scores)].reshape(1, 2)
        return waypoint
