import numpy as np
import pytest

from pypolo.planners import MyopicPlanner


@pytest.fixture
def planner():
    workspace = [0.0, 0.0, 10.0, 10.0]
    objective = lambda x: np.sum(np.power(x - [5, 5], 2), axis=1)
    num_candidates = 1000
    return MyopicPlanner(workspace, objective, num_candidates)


def test_plan(planner):
    state = np.array([0, 0, 0])
    waypoint = planner.plan(state)
    assert isinstance(waypoint, np.ndarray)
    assert waypoint.shape == (1, 2)
    assert np.all(waypoint >= planner.workspace[:2])
    assert np.all(waypoint <= planner.workspace[2:])
